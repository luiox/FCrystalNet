package com.github.luiox.fcrystalnet.poc.golden;

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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The Java twin of {@code read_frames.decode}: one .ncds v2 frame payload in, the decoded
 * fields out - plus {@link #toFrame}, which reshapes a decoded frame into the
 * {@link MinimalPerception.Frame} the live perception produces, so the golden test runs
 * {@code FeatureBuilder} exactly the way the game runs it. Test-only: no Minecraft types.
 */
final class FrameDecoder {

    static final int KIND_SELF = 0;
    static final int KIND_TARGET = 1;
    static final int KIND_OTHER = 2;
    static final int KIND_CRYSTAL = 3;

    /** One entity record; stats = health, maxHealth, armor, toughness, protection, resistance. */
    record Ent(int kind, Vec3d pos, Aabb box, float[] stats, List<Vec3d> preds) {
    }

    record Decoded(int schema, long seed, int preset, int difficulty, int aimSteps,
                   float[] tvel, float[] svel,
                   float tArmor, float tTough, float tProt, float tHealth,
                   float sArmor, float sTough, float sProt, float sHealth,
                   Vec3i origin, int sx, int sy, int sz, List<Ent> ents,
                   byte[] kind, byte[] valid, byte[] reach, byte[] breakable,
                   float[] dmg, float[] selfDmg) {
    }

    private FrameDecoder() {
    }

    /** Decodes one payload; the layout follows docs/dataset-schema.md. */
    static Decoded decode(byte[] payload) {
        ByteBuffer p = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        long seed = p.getLong();
        int preset = p.get() & 0xFF;
        int difficulty = p.get() & 0xFF;
        int aimSteps = p.get() & 0xFF;
        int entCount = p.get() & 0xFF;
        float[] tvel = float3(p);
        float[] svel = float3(p);
        float tArmor = p.getFloat();
        float tTough = p.getFloat();
        float tProt = p.getFloat();
        float tHealth = p.getFloat();
        float sArmor = p.getFloat();
        float sTough = p.getFloat();
        float sProt = p.getFloat();
        float sHealth = p.getFloat();
        Vec3i origin = new Vec3i(p.getInt(), p.getInt(), p.getInt());
        int sx = p.get() & 0xFF;
        int sy = p.get() & 0xFF;
        int sz = p.get() & 0xFF;
        int cells = sx * sy * sz;

        List<Ent> ents = new ArrayList<>(entCount);
        for (int i = 0; i < entCount; i++) {
            int kind = p.get() & 0xFF;
            Vec3d pos = new Vec3d(p.getDouble(), p.getDouble(), p.getDouble());
            Aabb box = new Aabb(p.getDouble(), p.getDouble(), p.getDouble(),
                    p.getDouble(), p.getDouble(), p.getDouble());
            float[] stats = new float[6];
            for (int s = 0; s < 6; s++) {
                stats[s] = p.getFloat();
            }
            int npred = p.get() & 0xFF;
            List<Vec3d> preds = new ArrayList<>(npred);
            for (int s = 0; s < npred; s++) {
                preds.add(new Vec3d(p.getDouble(), p.getDouble(), p.getDouble()));
            }
            ents.add(new Ent(kind, pos, box, stats, preds));
        }

        byte[] kind = u8(p, cells);
        byte[] valid = u8(p, cells);
        byte[] needsBase = u8(p, cells);
        byte[] reach = u8(p, cells);
        int schema = 1;
        byte[] breakable;
        if (payload.length > p.position()) {
            schema = 2;
            breakable = u8(p, cells);
        } else {
            breakable = new byte[cells];
        }
        float[] dmg = f32(p, cells);
        float[] selfDmg = f32(p, cells);
        if (p.position() != payload.length) {
            throw new IllegalStateException("frame payload over/under-run: "
                    + p.position() + " != " + payload.length);
        }
        return new Decoded(schema, seed, preset, difficulty, aimSteps, tvel, svel,
                tArmor, tTough, tProt, tHealth, sArmor, sTough, sProt, sHealth,
                origin, sx, sy, sz, ents, kind, valid, reach, breakable, dmg, selfDmg);
    }

    private static float[] float3(ByteBuffer p) {
        return new float[]{p.getFloat(), p.getFloat(), p.getFloat()};
    }

    private static byte[] u8(ByteBuffer p, int cells) {
        byte[] out = new byte[cells];
        p.get(out);
        return out;
    }

    private static float[] f32(ByteBuffer p, int cells) {
        float[] out = new float[cells];
        for (int i = 0; i < cells; i++) {
            out[i] = p.getFloat();
        }
        return out;
    }

    // --- reshape into the live perception's frame ---------------------------------------------

    /**
     * The frame as the game would have perceived it: the decoded kind bytes back into a
     * region, ents[0] into a self, ents[1] into the target, ents[1..] into the entity
     * list. Ids: target 1, others 2.. - internal to the golden test only.
     */
    static MinimalPerception.Frame toFrame(Decoded d) {
        BlockSnapshot[] blocks = new BlockSnapshot[d.sx() * d.sy() * d.sz()];
        for (int i = 0; i < blocks.length; i++) {
            blocks[i] = switch (d.kind()[i] & 0xFF) {
                case 1 -> BlockSnapshot.REPLACEABLE;
                case 2 -> BlockSnapshot.OBSIDIAN;
                case 3 -> BlockSnapshot.BEDROCK;
                case 4 -> BlockSnapshot.OTHER;
                default -> BlockSnapshot.AIR;
            };
        }
        WorldRegionSnapshot region = WorldRegionSnapshot.of(d.origin(), d.sx(), d.sy(),
                d.sz(), blocks, Map.of(),
                WorldRegionSnapshot.Difficulty.values()[d.difficulty()]);

        Ent self = d.ents().get(0);
        Ent target = d.ents().get(1);
        List<EntitySnapshot> entities = new ArrayList<>(d.ents().size() - 1);
        for (int i = 1; i < d.ents().size(); i++) {
            entities.add(entity(d.ents().get(i), i));
        }

        // The data has a self with eyeHeight 1.62 (a sneaking-height self) and a
        // fixed 0.6x1.8 box; the frame stores the box but not the eye, so the
        // reconstruction hardcodes the same constant. In-game the real eye height flows in
        // and FeatureBuilder's reach formula runs unchanged.
        float eyeHeight = 1.62f;
        SelfSnapshot selfSnapshot = new SelfSnapshot(self.pos(), eyeHeight,
                new ItemStackSnapshot[9], 0, new ItemStackSnapshot[27],
                new ItemStackSnapshot[4], ItemStackSnapshot.EMPTY,
                self.stats()[0], self.stats()[1], self.stats()[2], self.stats()[3],
                self.stats()[4], false, true, 0.0f, 0.0f, self.preds());
        TargetSnapshot targetSnapshot = new TargetSnapshot(entity(target, 1),
                target.preds(), true);
        // the dataset predicts at integer ticks, steps per frame; the live perception at
        // half ticks - the frame carries its own convention.
        return new MinimalPerception.Frame(selfSnapshot, targetSnapshot, entities, region,
                d.aimSteps(), 1.0);
    }

    private static EntitySnapshot entity(Ent e, int id) {
        String type = switch (e.kind()) {
            case KIND_CRYSTAL -> "minecraft:end_crystal";
            case KIND_TARGET -> "test:target";
            default -> "test:entity";
        };
        return new EntitySnapshot(id, type, e.pos(), e.box(),
                e.stats()[0], e.stats()[1], e.stats()[2], e.stats()[3], e.stats()[4],
                e.stats()[5], 0, false, e.preds());
    }
}
