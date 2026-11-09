package com.github.luiox.fcrystalnet.poc.features;

import com.github.luiox.fcrystalnet.poc.perception.MinimalPerception;
import com.github.luiox.fcrystalnet.core.math.Aabb;
import com.github.luiox.fcrystalnet.core.math.Vec3d;
import com.github.luiox.fcrystalnet.core.math.Vec3i;
import com.github.luiox.fcrystalnet.core.snapshot.EntitySnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.SelfSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.TargetSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.WorldRegionSnapshot;
import java.util.List;

/**
 * Frame tensors, the exact port of {@code ml/features.py} - the same 11 channels in the
 * same (channel, y, z, x) row-major order the ONNX session consumes. Keeping this a line
 * by line twin is the whole golden-test story: {@code FeatureGoldenTest} runs both on the
 * same decoded frames and fails on any drift.
 *
 * <p>Terrain keeps its raw five-kind identity as one-hot channels. Entity channels are
 * continuous distance fields {@code exp(-d/2)} to the entity box surface (1.0 inside,
 * smooth decay outside) instead of 0/1 rasters: the scored position lives between cells,
 * and the distance field keeps the sub-cell information the argmax needs for fine
 * ranking. Channel 9 is the reach map (self-eye to cell-centre &le; 4.5); channel 10 is
 * the existing-crystal distance field - the break side's subjects, whose damage is read
 * off the same dmg head (detonating a crystal at a cell is the cell's own explosion).
 *
 * <p>The crop never moves: it is target-centred by construction, 13x7x13 at the target's foot cell + (-6, -3, -6), so
 * "where is the target" is read off the aim channel directly.
 */
public final class FeatureBuilder {

    /** Crop extents, see docs/dataset-schema.md. */
    public static final int SX = 13;
    public static final int SY = 7;
    public static final int SZ = 13;
    public static final int CELLS = SX * SY * SZ;

    public static final int N_KIND = 5;
    public static final int CH_AIM = 5;
    public static final int CH_CUR = 6;
    public static final int CH_SELF = 7;
    public static final int CH_OTHER = 8;
    public static final int CH_REACH = 9;
    public static final int CH_CRYSTAL = 10;
    public static final int CHANNELS = 11;
    public static final int AUX_DIM = 14;
    public static final double SIGMA = 2.0;

    /** The interaction layer's placement gate, mirrored for the reach channel. */
    private static final double REACH = 4.5;

    private static final String CRYSTAL = "minecraft:end_crystal";

    private FeatureBuilder() {
    }

    /**
     * The crop origin for a target: foot cell - (6, 3, 6).
     */
    public static Vec3i originOf(TargetSnapshot target) {
        return target.current().pos().floor().offset(-6, -3, -6);
    }

    /** Crop-local row-major index, matching {@code WorldRegionSnapshot.index}'s order. */
    public static int idx(int x, int y, int z) {
        return y * SZ * SX + z * SX + x;
    }

    /** Crop-local row-major index of a world cell. */
    public static int idx(Vec3i origin, Vec3i cell) {
        return idx(cell.x() - origin.x(), cell.y() - origin.y(), cell.z() - origin.z());
    }

    /**
     * The model's terrain+entity tensor: {@code float[CHANNELS * CELLS]}, channel-major,
     * cells row-major (y, z, x). No float16 rounding here - the training pipeline rounds
     * to fp16 after building, and the golden test compares against exactly that, so this
     * stays double-precision and lets the tolerance absorb the rounding.
     */
    public static float[] buildX(MinimalPerception.Frame frame) {
        WorldRegionSnapshot region = frame.region();
        TargetSnapshot target = frame.target();
        SelfSnapshot self = frame.self();
        List<EntitySnapshot> entities = frame.entities();
        Vec3i origin = originOf(target);

        float[] x = new float[CHANNELS * CELLS];

        // Channels 0-4: the terrain kind one-hots over the crop.
        for (int y = 0; y < SY; y++) {
            for (int z = 0; z < SZ; z++) {
                for (int cx = 0; cx < SX; cx++) {
                    int kind = region.at(origin.offset(cx, y, z)).kind().ordinal();
                    if (kind < N_KIND) {
                        x[kind * CELLS + idx(cx, y, z)] = 1.0f;
                    }
                }
            }
        }

        // Channel 7: the self box (the data uses the same 0.6x1.8 box for ents[0]).
        float[] selfField = distField(origin, Aabb.around(self.pos(), 0.6, 1.8));
        System.arraycopy(selfField, 0, x, CH_SELF * CELLS, CELLS);

        // Channel 6: the target box now.
        EntitySnapshot tgt = target.current();
        System.arraycopy(distField(origin, tgt.box()), 0, x, CH_CUR * CELLS, CELLS);

        // Channel 5: the target box at the scored (predicted) position - the aim box is
        // the current box translated by the last prediction's offset.
        if (!target.predicted().isEmpty()) {
            Vec3d last = target.predicted().get(target.predicted().size() - 1);
            Vec3d shift = last.offset(-tgt.pos().x(), -tgt.pos().y(), -tgt.pos().z());
            Aabb aimBox = tgt.box().offset(shift.x(), shift.y(), shift.z());
            System.arraycopy(distField(origin, aimBox), 0, x, CH_AIM * CELLS, CELLS);
        }

        // Channels 8/10: nearest-other and crystal distance fields, max-composed.
        float[] other = null;
        float[] crystal = null;
        for (EntitySnapshot entity : entities) {
            boolean isCrystal = CRYSTAL.equals(entity.type());
            if (!isCrystal && entity.id() == tgt.id()) {
                continue; // the target is ents[1] in the data, never part of ents[2:]
            }
            float[] field = distField(origin, entity.box());
            if (isCrystal) {
                crystal = maxInto(crystal, field);
            } else {
                other = maxInto(other, field);
            }
        }
        if (other != null) {
            System.arraycopy(other, 0, x, CH_OTHER * CELLS, CELLS);
        }
        if (crystal != null) {
            System.arraycopy(crystal, 0, x, CH_CRYSTAL * CELLS, CELLS);
        }

        // Channel 9: the reach map, the interaction layer's exact gate.
        Vec3d eye = self.eyePos();
        double reachSq = REACH * REACH;
        for (int y = 0; y < SY; y++) {
            for (int z = 0; z < SZ; z++) {
                for (int cx = 0; cx < SX; cx++) {
                    Vec3i cell = origin.offset(cx, y, z);
                    Vec3d centre = new Vec3d(cell.x() + 0.5, cell.y() + 0.5, cell.z() + 0.5);
                    if (centre.distanceSquaredTo(eye) <= reachSq) {
                        x[CH_REACH * CELLS + idx(cx, y, z)] = 1.0f;
                    }
                }
            }
        }
        return x;
    }

    /**
     * {@code exp(-d / sigma)} per voxel centre, d = distance to the box surface
     * (0 inside) - features.py's {@code _dist_field}.
     */
    private static float[] distField(Vec3i origin, Aabb box) {
        float[] out = new float[CELLS];
        for (int y = 0; y < SY; y++) {
            double cy = origin.y() + y + 0.5;
            double dy = Math.max(Math.max(box.minY() - cy, cy - box.maxY()), 0.0);
            for (int z = 0; z < SZ; z++) {
                double cz = origin.z() + z + 0.5;
                double dz = Math.max(Math.max(box.minZ() - cz, cz - box.maxZ()), 0.0);
                double dyz = dy * dy + dz * dz;
                for (int cx = 0; cx < SX; cx++) {
                    double cxx = origin.x() + cx + 0.5;
                    double dx = Math.max(Math.max(box.minX() - cxx, cxx - box.maxX()), 0.0);
                    double d = Math.sqrt(dx * dx + dyz);
                    out[idx(cx, y, z)] = (float) Math.exp(-d / SIGMA);
                }
            }
        }
        return out;
    }

    private static float[] maxInto(float[] acc, float[] field) {
        if (acc == null) {
            return field.clone();
        }
        for (int i = 0; i < CELLS; i++) {
            acc[i] = Math.max(acc[i], field[i]);
        }
        return acc;
    }

    /**
     * The model's aux vector: {@code float[AUX_DIM]} in features.py's exact order and
     * normalisation. Velocities are per-tick deltas, recovered from the prediction list
     * the way the data wrote it ({@code preds[i] = pos + vel * step * stepTicks}).
     */
    public static float[] buildAux(MinimalPerception.Frame frame) {
        TargetSnapshot target = frame.target();
        SelfSnapshot self = frame.self();

        Vec3d tvel = velocityOf(target.current().pos(), target.predicted(),
                frame.aimSteps(), frame.stepTicks());
        Vec3d svel = velocityOf(self.pos(), self.predicted(),
                frame.aimSteps(), frame.stepTicks());
        return new float[]{
                (float) (tvel.x() / 0.35), (float) (tvel.y() / 0.35), (float) (tvel.z() / 0.35),
                (float) (svel.x() / 0.35), (float) (svel.y() / 0.35), (float) (svel.z() / 0.35),
                frame.aimSteps() / 5.0f,
                target.current().armor() / 20.0f,
                target.current().armorToughness() / 20.0f,
                target.current().protection() / 20.0f,
                target.current().health() / 20.0f,
                self.armor() / 20.0f,
                self.protection() / 20.0f,
                frame.region().difficulty().ordinal() / 3.0f,
        };
    }

    /**
     * Per-tick velocity recovered from a prediction list; zero when prediction is off.
     * The span is the frame's own convention ({@code aimSteps * stepTicks}): the dataset writes {@code preds[i] = pos + vel * i} at integer ticks, the live perception at
     * half ticks - both must recover the same per-tick velocity.
     */
    private static Vec3d velocityOf(Vec3d base, List<Vec3d> predicted, int aimSteps,
                                    double stepTicks) {
        if (predicted.isEmpty() || base == null || aimSteps <= 0) {
            return Vec3d.ZERO;
        }
        double span = aimSteps * stepTicks;
        Vec3d last = predicted.get(predicted.size() - 1);
        return new Vec3d((last.x() - base.x()) / span, (last.y() - base.y()) / span,
                (last.z() - base.z()) / span);
    }
}
