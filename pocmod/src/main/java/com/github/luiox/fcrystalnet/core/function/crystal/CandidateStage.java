package com.github.luiox.fcrystalnet.core.function.crystal;

import com.github.luiox.fcrystalnet.core.function.crystal.CrystalCandidate;
import com.github.luiox.fcrystalnet.core.math.Vec3i;
import com.github.luiox.fcrystalnet.core.math.Aabb;
import com.github.luiox.fcrystalnet.core.snapshot.BlockSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.EntitySnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.SelfSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.TargetSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.WorldRegionSnapshot;
import java.util.ArrayList;
import java.util.List;

/**
 * The candidate stage: target-centred enumeration of every cell a crystal could occupy,
 * with the base union folded in - "search depth = how many obsidian we place" reduced to
 * one {@code blocksNeeded} field per candidate.
 *
 * <p>Also the break side's source: the crystal entities worth attacking, extracted from
 * the same entity list. One stage owns both, so a future spider-web or respawn-anchor
 * variant swaps this stage without touching anything downstream.
 */
public final class CandidateStage {

    /** Horizontal candidate search radius around the target's cell. */
    public static final int RADIUS = 4;
    /** Crystal cells sit at the target's feet height minus two up to head height plus one. */
    public static final int Y_MIN = -2;
    public static final int Y_MAX = 2;

    /**
     * How much a candidate cell's keep-out zone grows beyond each entity's box. The
     * server refuses a crystal whose cell's two-high box touches any entity ({@code
     * EndCrystalItem.useOn} checks {@code getEntities}), and the decision runs on a
     * snapshot up to two ticks old - a circling target moves a tenth of a block per tick,
     * so a candidate that barely cleared at snapshot time fails at dispatch.
     */
    public static final double BLOCKER_MARGIN = 0.25;

    private static final String CRYSTAL = "minecraft:end_crystal";
    /** How far from the target a breakable crystal still counts as part of the fight. */
    private static final double BREAK_RANGE = 12.0;

    private CandidateStage() {
    }

    /**
     * Every crystal cell the region says is possible: an air cell over an existing base,
     * or over one placeable base - the base-union the later stages argue over.
     *
     * <p>The rules are the 1.21.11 {@code EndCrystalItem.useOn} accept test, bytecode
     * verified: the support block must be obsidian or bedrock, the crystal cell above it
     * must pass {@code Level.isEmptyBlock} - which is {@code state.isAir()}, strictly, so
     * a replaceable cell (grass, snow) is <em>not</em> placeable however harmless it
     * looks - and the cell's two-high footprint must hold no entity at all
     * ({@code getEntities}). This stage enforces all three, plus a keep-out margin for
     * snapshot drift, so a candidate that survives here is one the server would accept.
     *
     * <p>A needs-base candidate plays the same test one level up: the obsidian goes into
     * a cell that accepts a placement (air or replaceable), on top of any real block -
     * block placement clicks any non-empty block's face, obsidian or not - with no
     * entity occupying the cell (an obsidian has collision, so a body in the cell
     * refuses it exactly as it refuses a crystal).
     *
     * @param allowBase the base gate: off means existing-base candidates only, and no
     *                  needs-base candidate exists anywhere downstream
     */
    public static List<CrystalCandidate> enumerate(WorldRegionSnapshot region, TargetSnapshot target,
                                                   List<EntitySnapshot> entities, SelfSnapshot self,
                                                   boolean allowBase) {
        Vec3i centre = target.current().pos().floor();
        List<CrystalCandidate> candidates = new ArrayList<>(32);
        List<Aabb> blockers = blockers(entities, self);
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            for (int dz = -RADIUS; dz <= RADIUS; dz++) {
                for (int dy = Y_MIN; dy <= Y_MAX; dy++) {
                    Vec3i cell = centre.offset(dx, dy, dz);
                    // isEmptyBlock == isAir: the replaceable cells a block placement
                    // could overwrite are not cells a crystal may occupy.
                    if (!region.contains(cell) || region.at(cell).kind() != BlockSnapshot.Kind.AIR) {
                        continue;
                    }
                    if (intersectsAny(crystalFootprint(cell), blockers)) {
                        continue;
                    }
                    Vec3i below = cell.below();
                    if (region.at(below).isBase()) {
                        candidates.add(CrystalCandidate.onExistingBase(cell));
                    } else if (allowBase
                            && region.at(below).isEmpty()
                            && region.contains(below.below())
                            && !region.at(below.below()).isEmpty()
                            && !intersectsAny(cellBox(below), blockers)) {
                        candidates.add(CrystalCandidate.needsBase(cell, below));
                    }
                }
            }
        }
        return candidates;
    }

    /** The crystal entities within fight range of the target, nearest-order preserved. */
    public static List<EntitySnapshot> breakables(TargetSnapshot target, List<EntitySnapshot> entities) {
        List<EntitySnapshot> out = new ArrayList<>(4);
        for (EntitySnapshot entity : entities) {
            if (!CRYSTAL.equals(entity.type())) {
                continue;
            }
            if (entity.pos().distanceTo(target.current().pos()) > BREAK_RANGE) {
                continue;
            }
            out.add(entity);
        }
        return out;
    }

    /** The boxes a placement must not touch: every nearby entity plus the local player. */
    private static List<Aabb> blockers(List<EntitySnapshot> entities, SelfSnapshot self) {
        List<Aabb> boxes = new ArrayList<>(entities.size() + 1);
        for (EntitySnapshot entity : entities) {
            boxes.add(entity.box().inflate(BLOCKER_MARGIN));
        }
        if (self != null) {
            boxes.add(Aabb.around(self.pos(), 0.6, 1.8).inflate(BLOCKER_MARGIN));
        }
        return boxes;
    }

    /** The server's crystal accept test uses the cell's footprint extended two blocks up. */
    private static Aabb crystalFootprint(Vec3i cell) {
        return new Aabb(cell.x(), cell.y(), cell.z(),
                cell.x() + 1.0, cell.y() + 2.0, cell.z() + 1.0);
    }

    private static boolean intersectsAny(Aabb box, List<Aabb> blockers) {
        for (Aabb blocker : blockers) {
            if (box.intersects(blocker)) {
                return true;
            }
        }
        return false;
    }

    private static Aabb cellBox(Vec3i cell) {
        return new Aabb(cell.x(), cell.y(), cell.z(),
                cell.x() + 1.0, cell.y() + 1.0, cell.z() + 1.0);
    }
}
