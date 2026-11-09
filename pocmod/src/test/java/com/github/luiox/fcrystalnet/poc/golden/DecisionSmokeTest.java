package com.github.luiox.fcrystalnet.poc.golden;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.luiox.fcrystalnet.poc.decide.PocDecision;
import com.github.luiox.fcrystalnet.poc.features.FeatureBuilder;
import com.github.luiox.fcrystalnet.poc.infer.CrystalNetBridge;
import com.github.luiox.fcrystalnet.poc.perception.MinimalPerception;
import com.github.luiox.fcrystalnet.core.math.Aabb;
import com.github.luiox.fcrystalnet.core.math.Vec3d;
import com.github.luiox.fcrystalnet.core.math.Vec3i;
import com.github.luiox.fcrystalnet.core.snapshot.BlockSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.EntitySnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.ItemStackSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.SelfSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.TargetSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.WorldRegionSnapshot;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * The in-game decision loop without Minecraft: a synthetic scene that mirrors the
 * acceptance setup exactly - a static fake player standing on an obsidian pad, the local
 * player a few blocks away, grass floor - run through FeatureBuilder, the real DLL and
 * PocDecision. The live perception produces the same shapes, so a place pick here but
 * not in-game indicts the perception reads; no pick here indicts the scene or the gates,
 * and the printed stats say which.
 */
class DecisionSmokeTest {

    @Test
    void syntheticPadSceneProducesPlacePick() throws Exception {
        CrystalNetBridge acquired = null;
        Throwable failure = null;
        try {
            acquired = CrystalNetBridge.acquire();
        } catch (Throwable t) {
            failure = t;
        }
        assumeTrue(acquired != null, "native inference unavailable: " + failure);
        try (CrystalNetBridge bridge = acquired) {
            // The acceptance scene: obsidian pad at y=70, fake player on top at
            // (252, 71, 258), the local player ~4.5 blocks away on the grass. The pad is
            // 5x5 on purpose: enumerate's blocker margin (0.25) makes a centred target's
            // keep-out poke into every adjacent cell, so a 3x3 pad yields zero candidates
            // by design - a real setup puts obsidian two cells out too.
            Vec3i padCell = new Vec3i(252, 70, 258);
            Vec3d targetPos = new Vec3d(padCell.x() + 0.5, padCell.y() + 1.0,
                    padCell.z() + 0.5);

            int rx = 12;
            int ry0 = 65;
            Vec3i min = new Vec3i(padCell.x() - rx, ry0, padCell.z() - rx);
            int sx = 25;
            int sy = 16;
            int sz = 25;
            BlockSnapshot[] blocks = new BlockSnapshot[sx * sy * sz];
            for (int y = 0; y < sy; y++) {
                for (int z = 0; z < sz; z++) {
                    for (int x = 0; x < sx; x++) {
                        Vec3i cell = new Vec3i(min.x() + x, ry0 + y, min.z() + z);
                        BlockSnapshot block = BlockSnapshot.AIR;
                        if (cell.y() == 70) {
                            // grass floor, obsidian under and around the target
                            block = Math.abs(cell.x() - padCell.x()) <= 2
                                    && Math.abs(cell.z() - padCell.z()) <= 2
                                    ? BlockSnapshot.OBSIDIAN : BlockSnapshot.OTHER;
                        } else if (cell.y() < 70) {
                            block = BlockSnapshot.OTHER; // solid underground
                        }
                        blocks[x + z * sx + y * sx * sz] = block;
                    }
                }
            }
            WorldRegionSnapshot region = WorldRegionSnapshot.of(min, sx, sy, sz, blocks,
                    Map.of(), WorldRegionSnapshot.Difficulty.NORMAL);

            EntitySnapshot target = EntitySnapshot.unpredicted(42, "minecraft:player",
                    targetPos, Aabb.around(targetPos, 0.6, 1.8),
                    20.0f, 20.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, false);
            TargetSnapshot targetSnapshot = new TargetSnapshot(target, List.of(), true);

            Vec3d selfPos = new Vec3d(padCell.x() - 3.5, padCell.y() + 1.0,
                    padCell.z() + 1.5);
            // A crystal-pvp loadout: blast protection caps the self damage inside reach,
            // which is exactly how a real player survives their own crystals.
            SelfSnapshot self = SelfSnapshot.of(selfPos, 1.8,
                    new ItemStackSnapshot[]{ItemStackSnapshot.EMPTY}, 0,
                    new ItemStackSnapshot[0], new ItemStackSnapshot[0],
                    ItemStackSnapshot.EMPTY, 20.0f, 20.0f, 0.0f, 0.0f, 20.0f,
                    false, true, 0.0f, 0.0f);
            List<EntitySnapshot> entities = List.of(target);

            MinimalPerception.Frame frame = new MinimalPerception.Frame(self,
                    targetSnapshot, entities, region, 4, 0.5);

            System.out.printf("enumerate: %d candidates%n",
                    com.github.luiox.fcrystalnet.core.function.crystal.CandidateStage.enumerate(
                            frame.region(), frame.target(), frame.entities(), frame.self(),
                            false).size());

            float[] x = FeatureBuilder.buildX(frame);
            float[] aux = FeatureBuilder.buildAux(frame);
            // Channel stats against golden frame 0 - the model's dmg field collapsing to
            // ~1.5 while the oracle sees 70+ means an input channel is off-distribution;
            // the per-channel max/mean pair says which one.
            List<GoldenFrames.Frame> golden = GoldenFrames.read(
                    java.nio.file.Path.of("src", "test", "resources", "golden.bin"));
            float[] gx = golden.get(0).x();
            System.out.println("channel stats (synthetic vs golden0): [ch] max/mean");
            for (int c = 0; c < FeatureBuilder.CHANNELS; c++) {
                float smax = 0, ssum = 0, gmax = 0, gsum = 0;
                for (int i = 0; i < FeatureBuilder.CELLS; i++) {
                    float s = x[c * FeatureBuilder.CELLS + i];
                    float g = gx[c * FeatureBuilder.CELLS + i];
                    smax = Math.max(smax, s);
                    ssum += s;
                    gmax = Math.max(gmax, g);
                    gsum += g;
                }
                System.out.printf("  ch%02d: %.3f/%.4f  vs  %.3f/%.4f%n", c, smax,
                        ssum / FeatureBuilder.CELLS, gmax, gsum / FeatureBuilder.CELLS);
            }
            System.out.print("aux synthetic: ");
            for (float a : aux) {
                System.out.printf("%.3f ", a);
            }
            System.out.print("\naux golden0:   ");
            for (float a : golden.get(0).aux()) {
                System.out.printf("%.3f ", a);
            }
            System.out.println();

            float[] dmg = new float[FeatureBuilder.CELLS];
            float[] valid = new float[FeatureBuilder.CELLS];
            bridge.infer(x, aux, dmg, valid);

            // Teacher vs model on the same cells: the target's own cell and the pad
            // diagonal - the teacher (DamageModel) is the label the model was trained on.
            Vec3i smokeOrigin = FeatureBuilder.originOf(targetSnapshot);
            int[][] probes = {{6, 3, 6}, {5, 3, 5}, {7, 3, 7}, {7, 3, 5}};
            for (int[] p : probes) {
                int idx = FeatureBuilder.idx(p[0], p[1], p[2]);
                Vec3i cell = smokeOrigin.offset(p[0], p[1], p[2]);
                double teacher = com.github.luiox.fcrystalnet.core.function.crystal.DamageModel
                        .explosionDamage(region, Vec3d.centered(cell), target,
                                targetSnapshot.aimPos());
                System.out.printf("probe crop%s world%s: teacher=%.2f model=%.2f validLogit=%.3f%n",
                        java.util.Arrays.toString(p), cell, teacher, dmg[idx], valid[idx]);
            }

            Vec3i origin = FeatureBuilder.originOf(targetSnapshot);
            float maxDmg = Float.NEGATIVE_INFINITY;
            float maxValid = Float.NEGATIVE_INFINITY;
            int argmaxDmg = -1;
            for (int i = 0; i < FeatureBuilder.CELLS; i++) {
                if (dmg[i] > maxDmg) {
                    maxDmg = dmg[i];
                    argmaxDmg = i;
                }
                if (valid[i] > maxValid) {
                    maxValid = valid[i];
                }
            }
            int dx = argmaxDmg % FeatureBuilder.SX;
            int dz = (argmaxDmg / FeatureBuilder.SX) % FeatureBuilder.SZ;
            int dy = argmaxDmg / (FeatureBuilder.SX * FeatureBuilder.SZ);
            System.out.printf("model: maxDmg=%.3f @crop(%d,%d,%d) maxValid(logit)=%.3f%n",
                    maxDmg, dx, dy, dz, maxValid);

            PocDecision.Decision decision = PocDecision.decide(frame, dmg, valid);
            System.out.printf("decision: place=%s%n", decision.place());
            System.out.printf("oracle:   placeExact=%s%n", decision.placeExact());

            assertTrue(decision.placeExact() != null,
                    "oracle found nothing performable - scene or gates broken");
            assertTrue(decision.place() != null,
                    "model produced no pick over a performable set of "
                            + "exact-dmg-70 cells");
        }
    }
}
