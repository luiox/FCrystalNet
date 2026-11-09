package com.github.luiox.fcrystalnet.poc.decide;

import com.github.luiox.fcrystalnet.poc.features.FeatureBuilder;
import com.github.luiox.fcrystalnet.poc.perception.MinimalPerception;
import com.github.luiox.fcrystalnet.core.function.crystal.CandidateStage;
import com.github.luiox.fcrystalnet.core.function.crystal.CrystalCandidate;
import com.github.luiox.fcrystalnet.core.function.crystal.DamageModel;
import com.github.luiox.fcrystalnet.core.math.Aabb;
import com.github.luiox.fcrystalnet.core.math.Vec3d;
import com.github.luiox.fcrystalnet.core.math.Vec3i;
import com.github.luiox.fcrystalnet.core.snapshot.EntitySnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.SelfSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.WorldRegionSnapshot;
import java.util.List;

/**
 * The POC's decision: the model proposes, the exact model disposes. Placement candidates
 * come from {@link CandidateStage#enumerate} (the teacher's own enumeration, allowBase
 * off to match the v2 dataset) and the exact gates - the vanilla placement tests already
 * inside enumerate, reach, and the {@link DamageModel} self-veto - decide what is
 * performable. The model's dmg field then only <em>ranks</em> that set: the head is
 * trained per-frame normalized ({@code train.py norm_dmg} - it learns the field's shape,
 * the argmax order, not absolute half-hearts), so its values are a score, never a
 * damage. The pick's absolute damage comes from the exact re-rank, and the action floor
 * applies there.
 *
 * <p>Break is the same machinery through the shared damage head: a crystal detonated at
 * a cell is that cell's own explosion, so breakables rank by the model damage at their
 * own cells. The same cycle also computes the exact argmax (the teacher's own answer)
 * over the same performable set, so the driver can log the live regret of following the
 * model instead of the oracle.
 */
public final class PocDecision {

    /** The action floor, in exact half-hearts at the picked cell (the model's own
     *  output has no absolute scale - see the class javadoc). */
    public static final double MIN_ACTION_DMG = 2.0;
    /** The self-veto margin: an explosion that leaves one heart of slack is allowed. */
    public static final double SELF_VETO_MARGIN = 1.0;
    /** Vanilla's attack reach; a break pick beyond this cannot be executed. */
    public static final double BREAK_ATTACK_REACH = 3.0;
    /** The dataset reach gate both sides share (the placement map). */
    private static final double REACH = 4.5;

    private static final String CRYSTAL = "minecraft:end_crystal";

    private PocDecision() {
    }

    /** One performable placement the model picked. */
    public record PlaceAction(CrystalCandidate candidate, double modelDmg, double exactDmg) {
    }

    /** One performable break the model picked. */
    public record BreakAction(EntitySnapshot crystal, Vec3i cell, double modelDmg,
                              double exactDmg) {
    }

    /**
     * Both sides of one cycle. {@code place}/{@code break} are the model's picks
     * (null = nothing passed the gates); {@code placeExact}/{@code breakExact} are the
     * exact argmaxes over the same performable sets - the oracle the regret is measured
     * against. All four are nullable independently.
     */
    public record Decision(PlaceAction place, PlaceAction placeExact,
                           BreakAction breakAction, BreakAction breakExact) {

        /** {@code exactOfChosen - modelOfChosen} on the side that acted; 0 when nothing did. */
        public double regret(boolean placeSide) {
            if (placeSide) {
                return this.place == null || this.placeExact == null
                        ? 0.0 : this.placeExact.exactDmg() - this.place.exactDmg();
            }
            return this.breakAction == null || this.breakExact == null
                    ? 0.0 : this.breakExact.exactDmg() - this.breakAction.exactDmg();
        }
    }

    /**
     * One decision cycle over a perception frame plus the model's output maps. The
     * valid map is unused here - validity is exact (enumerate, reach, self-veto); the
     * model's job is the ranking.
     */
    public static Decision decide(MinimalPerception.Frame frame, float[] dmg, float[] valid) {
        List<CrystalCandidate> placeCandidates = CandidateStage.enumerate(frame.region(),
                frame.target(), frame.entities(), frame.self(), false);
        PlaceAction place = placePick(frame, placeCandidates, dmg);
        PlaceAction placeExact = exactPick(frame, placeCandidates, dmg);

        List<EntitySnapshot> breakCandidates = CandidateStage.breakables(frame.target(),
                frame.entities());
        BreakAction breakAction = breakPick(frame, breakCandidates, dmg);
        BreakAction breakExact = breakExactPick(frame, breakCandidates, dmg);

        return new Decision(place, placeExact, breakAction, breakExact);
    }

    // --- place side -------------------------------------------------------------------------

    /**
     * The model's place pick: argmax by model score over the performable set (enumerate
     * &cap; reach &cap; exact self-veto), then the exact action floor on the pick.
     */
    private static PlaceAction placePick(MinimalPerception.Frame frame,
                                         List<CrystalCandidate> candidates,
                                         float[] dmg) {
        Vec3i origin = FeatureBuilder.originOf(frame.target());
        SelfSnapshot self = frame.self();
        EntitySnapshot selfEntity = selfEntity(self);
        double health = self.health();
        PlaceAction best = null;
        for (CrystalCandidate candidate : candidates) {
            int index = FeatureBuilder.idx(origin, candidate.pos());
            if (index < 0 || index >= FeatureBuilder.CELLS) {
                continue; // enumerate sits inside the crop by construction; belt only
            }
            if (!inReach(frame, candidate.pos())) {
                continue;
            }
            if (selfDmgAt(frame.region(), candidate.pos(), selfEntity, self)
                    >= health - SELF_VETO_MARGIN) {
                continue;
            }
            double score = dmg[index];
            if (best == null || score > best.modelDmg()) {
                best = new PlaceAction(candidate, score, exactDmgAt(frame, candidate.pos()));
            }
        }
        if (best != null && best.exactDmg() < MIN_ACTION_DMG) {
            return null; // the board is worthless - exact damage decides, not the score
        }
        return best;
    }

    /** The oracle's place pick: argmax by exact damage over the performable set. */
    private static PlaceAction exactPick(MinimalPerception.Frame frame,
                                         List<CrystalCandidate> candidates, float[] dmg) {
        Vec3i origin = FeatureBuilder.originOf(frame.target());
        SelfSnapshot self = frame.self();
        EntitySnapshot selfEntity = selfEntity(self);
        double health = self.health();
        PlaceAction best = null;
        for (CrystalCandidate candidate : candidates) {
            if (!inReach(frame, candidate.pos())) {
                continue;
            }
            if (selfDmgAt(frame.region(), candidate.pos(), selfEntity, self)
                    >= health - SELF_VETO_MARGIN) {
                continue;
            }
            double exact = exactDmgAt(frame, candidate.pos());
            if (best == null || exact > best.exactDmg()) {
                best = new PlaceAction(candidate, modelDmgAt(frame, candidate.pos(), dmg), exact);
            }
        }
        return best;
    }

    // --- break side -------------------------------------------------------------------------

    /**
     * The model's break pick: performable breakables (in-crop &cap; reach &cap; exact
     * self-veto) argmax by model score at the crystal's own cell, then the exact action
     * floor on the pick - the dataset's break gate (brk &amp; reach &amp; veto) with the
     * ranking delegated to the model.
     */
    private static BreakAction breakPick(MinimalPerception.Frame frame,
                                         List<EntitySnapshot> crystals, float[] dmg) {
        Vec3i origin = FeatureBuilder.originOf(frame.target());
        SelfSnapshot self = frame.self();
        EntitySnapshot selfEntity = selfEntity(self);
        double health = self.health();
        BreakAction best = null;
        for (EntitySnapshot crystal : crystals) {
            Vec3i cell = crystal.pos().floor();
            int index = FeatureBuilder.idx(origin, cell);
            if (index < 0 || index >= FeatureBuilder.CELLS) {
                continue; // out-of-crop crystals are dropped, the model never sees them
            }
            if (!inReach(frame, cell)) {
                continue;
            }
            if (selfDmgAt(frame.region(), cell, selfEntity, self)
                    >= health - SELF_VETO_MARGIN) {
                continue;
            }
            double score = dmg[index];
            if (best == null || score > best.modelDmg()) {
                best = new BreakAction(crystal, cell, score, exactDmgAt(frame, cell));
            }
        }
        if (best != null && best.exactDmg() < MIN_ACTION_DMG) {
            return null;
        }
        return best;
    }

    /** The oracle's break pick: argmax by exact damage over the performable set. */
    private static BreakAction breakExactPick(MinimalPerception.Frame frame,
                                              List<EntitySnapshot> crystals, float[] dmg) {
        Vec3i origin = FeatureBuilder.originOf(frame.target());
        SelfSnapshot self = frame.self();
        EntitySnapshot selfEntity = selfEntity(self);
        double health = self.health();
        BreakAction best = null;
        for (EntitySnapshot crystal : crystals) {
            Vec3i cell = crystal.pos().floor();
            int index = FeatureBuilder.idx(origin, cell);
            if (index < 0 || index >= FeatureBuilder.CELLS || !inReach(frame, cell)) {
                continue;
            }
            if (selfDmgAt(frame.region(), cell, selfEntity, self)
                    >= health - SELF_VETO_MARGIN) {
                continue;
            }
            double exact = exactDmgAt(frame, cell);
            if (best == null || exact > best.exactDmg()) {
                best = new BreakAction(crystal, cell, modelDmgAt(frame, cell, dmg), exact);
            }
        }
        return best;
    }

    // --- exact evaluation -------------------------------

    /**
     * Exact target damage of an explosion at {@code cell}'s bottom-centre, evaluated at
     * the target's aim position - the teacher's {@code dmg} value for the cell.
     */
    public static double exactDmgAt(MinimalPerception.Frame frame, Vec3i cell) {
        return DamageModel.explosionDamage(frame.region(), Vec3d.centered(cell),
                frame.target().current(), frame.target().aimPos());
    }

    /** The model's damage value at a world cell (0 outside the crop). */
    public static double modelDmgAt(MinimalPerception.Frame frame, Vec3i cell, float[] dmg) {
        int index = FeatureBuilder.idx(FeatureBuilder.originOf(frame.target()), cell);
        return index < 0 || index >= FeatureBuilder.CELLS ? 0.0 : dmg[index];
    }

    /**
     * Worst exact self damage over the current position and every predicted one - the
     * teacher's selfDmg loop (the hurtTime asymmetry takes the max).
     */
    public static double selfDmgAt(WorldRegionSnapshot region, Vec3i cell,
                                   EntitySnapshot selfEntity, SelfSnapshot self) {
        Vec3d source = Vec3d.centered(cell);
        double worst = DamageModel.explosionDamage(region, source, selfEntity, self.pos());
        for (Vec3d p : self.predicted()) {
            worst = Math.max(worst, DamageModel.explosionDamage(region, source, selfEntity, p));
        }
        return worst;
    }

    /** The dataset reach gate: eye to cell centre, within 4.5. */
    private static boolean inReach(MinimalPerception.Frame frame, Vec3i cell) {
        Vec3d centre = new Vec3d(cell.x() + 0.5, cell.y() + 0.5, cell.z() + 0.5);
        return centre.distanceSquaredTo(frame.self().eyePos()) <= REACH * REACH;
    }

    /** The self entity as an {@code EntitySnapshot}. */
    public static EntitySnapshot selfEntity(SelfSnapshot self) {
        return EntitySnapshot.unpredicted(-1, "self", self.pos(),
                Aabb.around(self.pos(), 0.6, 1.8),
                self.health(), 20.0f, self.armor(), self.armorToughness(), self.protection(),
                0.0f, 0, false);
    }
}
