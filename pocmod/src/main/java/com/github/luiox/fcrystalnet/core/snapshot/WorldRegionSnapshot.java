package com.github.luiox.fcrystalnet.core.snapshot;

import com.github.luiox.fcrystalnet.core.math.Vec3d;
import com.github.luiox.fcrystalnet.core.math.Vec3i;
import java.util.Map;

/**
 * An axis-aligned block region, stored as one flat array plus an immutable overlay of
 * virtual blocks.
 *
 * <p>This is the shape the damage model reads. Two decisions matter:
 *
 * <ul>
 *   <li><strong>Flat array, position arithmetic in ints.</strong> A 9x9x9 region is 729
 *       array slots; the whole snapshot is a few KB and stays cache-resident, and the
 *       worker thread never allocates while walking it.</li>
 *   <li><strong>The overlay is part of the snapshot.</strong> A hypothetical obsidian is
 *       a {@code Map} entry handed in at snapshot time - never global mutable state, so
 *       "base + pending mine + ignore terrain" compose without anyone's writes racing,
 *       and the same damage function answers "what if this block were obsidian" with no
 *       mixin anywhere.</li>
 * </ul>
 *
 * <p>Out-of-region reads return {@link BlockSnapshot#AIR}, which is the honest answer for
 * "we did not look there" only when callers keep their regions bigger than their
 * questions. The fabric implementation sizes the region from the modules' declared needs.
 */
public final class WorldRegionSnapshot {

    /**
     * The world difficulty the snapshot was taken under. Explosion damage scales with it
     * before the armour and effect reductions (EASY softens, HARD multiplies by 1.5), so
     * it belongs to the snapshot exactly like the blocks do - the decision answers "what
     * would happen in this world", and the difficulty is part of that world.
     */
    public enum Difficulty {
        PEACEFUL, EASY, NORMAL, HARD
    }

    private final Vec3i min;
    private final int sizeX;
    private final int sizeY;
    private final int sizeZ;
    private final BlockSnapshot[] blocks;
    private final Map<Vec3i, BlockSnapshot> overlay;
    private final Difficulty difficulty;

    private WorldRegionSnapshot(Vec3i min, int sizeX, int sizeY, int sizeZ,
                                BlockSnapshot[] blocks, Map<Vec3i, BlockSnapshot> overlay,
                                Difficulty difficulty) {
        this.min = min;
        this.sizeX = sizeX;
        this.sizeY = sizeY;
        this.sizeZ = sizeZ;
        this.blocks = blocks;
        this.overlay = overlay;
        this.difficulty = difficulty;
    }

    /** Builds a region from row-major blocks ({@code index(x,y,z)} order), NORMAL difficulty. */
    public static WorldRegionSnapshot of(Vec3i min, int sizeX, int sizeY, int sizeZ,
                                         BlockSnapshot[] blocks, Map<Vec3i, BlockSnapshot> overlay) {
        return of(min, sizeX, sizeY, sizeZ, blocks, overlay, Difficulty.NORMAL);
    }

    /** Builds a region from row-major blocks, with the world's difficulty attached. */
    public static WorldRegionSnapshot of(Vec3i min, int sizeX, int sizeY, int sizeZ,
                                         BlockSnapshot[] blocks, Map<Vec3i, BlockSnapshot> overlay,
                                         Difficulty difficulty) {
        if (blocks.length != sizeX * (long) sizeY * sizeZ) {
            throw new IllegalArgumentException("blocks length " + blocks.length
                    + " != " + sizeX + "*" + sizeY + "*" + sizeZ);
        }
        return new WorldRegionSnapshot(min, sizeX, sizeY, sizeZ, blocks,
                overlay == null ? Map.of() : Map.copyOf(overlay),
                difficulty == null ? Difficulty.NORMAL : difficulty);
    }

    /** A uniform region - flat floor scenarios in tests are one line. NORMAL difficulty. */
    public static WorldRegionSnapshot uniform(Vec3i min, int sizeX, int sizeY, int sizeZ,
                                              BlockSnapshot fill) {
        return uniform(min, sizeX, sizeY, sizeZ, fill, Difficulty.NORMAL);
    }

    /** A uniform region with the difficulty pinned, for the difficulty-scaling tests. */
    public static WorldRegionSnapshot uniform(Vec3i min, int sizeX, int sizeY, int sizeZ,
                                              BlockSnapshot fill, Difficulty difficulty) {
        BlockSnapshot[] blocks = new BlockSnapshot[sizeX * sizeY * sizeZ];
        java.util.Arrays.fill(blocks, fill);
        return new WorldRegionSnapshot(min, sizeX, sizeY, sizeZ, blocks, Map.of(), difficulty);
    }

    public Vec3i min() {
        return this.min;
    }

    public Vec3i max() {
        return new Vec3i(this.min.x() + this.sizeX - 1, this.min.y() + this.sizeY - 1,
                this.min.z() + this.sizeZ - 1);
    }

    public int sizeX() {
        return this.sizeX;
    }

    public int sizeY() {
        return this.sizeY;
    }

    public int sizeZ() {
        return this.sizeZ;
    }

    /** The overlay of virtual blocks; immutable, worker-thread readable. */
    public Map<Vec3i, BlockSnapshot> overlay() {
        return this.overlay;
    }

    /**
     * The same region with one more virtual block - the overlay mechanism the base
     * extension scores through: the decision hands {@code DamageStage} a region that
     * already contains the hypothetical obsidian, and the stage stays unaware that a
     * base was ever placed. Shares the block array; the copy is one map entry.
     */
    public WorldRegionSnapshot withOverlay(Vec3i pos, BlockSnapshot block) {
        java.util.Map<Vec3i, BlockSnapshot> extended = new java.util.HashMap<>(this.overlay);
        extended.put(pos, block);
        return new WorldRegionSnapshot(this.min, this.sizeX, this.sizeY, this.sizeZ,
                this.blocks, java.util.Map.copyOf(extended), this.difficulty);
    }

    /** The world difficulty this snapshot was taken under; drives explosion scaling. */
    public Difficulty difficulty() {
        return this.difficulty;
    }

    public boolean contains(Vec3i pos) {
        return pos.isWithin(this.min, this.max());
    }

    private int index(int x, int y, int z) {
        return (y - this.min.y()) * this.sizeZ * this.sizeX
                + (z - this.min.z()) * this.sizeX
                + (x - this.min.x());
    }

    /**
     * The block at {@code pos}: overlay first, then the array, {@code AIR} outside.
     *
     * <p>The overlay shadows the real world on purpose - a hypothetical obsidian must
     * participate in exposure and placement checks, or the damage model answers a
     * question about a world that no longer exists.
     */
    public BlockSnapshot at(Vec3i pos) {
        BlockSnapshot overridden = this.overlay.get(pos);
        if (overridden != null) {
            return overridden;
        }
        if (!this.contains(pos)) {
            return BlockSnapshot.AIR;
        }
        BlockSnapshot block = this.blocks[this.index(pos.x(), pos.y(), pos.z())];
        // A null cell means the builder did not fill it; "nothing" is the safe reading.
        return block == null ? BlockSnapshot.AIR : block;
    }

    public BlockSnapshot at(int x, int y, int z) {
        return this.at(new Vec3i(x, y, z));
    }

    /** Explosion resistance at {@code pos}, overlay included. */
    public float resistanceAt(Vec3i pos) {
        return this.at(pos).blastResistance();
    }

    /**
     * Whether a straight line from {@code from} to {@code to} is blocked by any block
     * with resistance above {@code minResistance}, sampling by voxel traversal. Overlay
     * blocks participate. Endpoints outside the region terminate the ray (treated as
     * unobstructed beyond the edge - the region is expected to cover the interesting
     * part).
     */
    public boolean occluded(Vec3d from, Vec3d to, float minResistance) {
        return this.traverse(from, to, false, minResistance);
    }

    /**
     * Whether a straight line from {@code from} to {@code to} is blocked by any block
     * with a collision shape - vanilla's {@code ClipContext.Block.COLLIDER}, which is
     * what {@code getSeenPercent} clips with (verified in the 1.21.11 bytecode). Every
     * solid block occludes; air and replaceable cells (fluids included) do not. Overlay
     * blocks participate, so a hypothetical wall casts a real shadow.
     */
    public boolean occludedByCollision(Vec3d from, Vec3d to) {
        return this.traverse(from, to, true, 0.0f);
    }

    /** The one voxel traversal; {@code solidOnly} switches between the two occluder rules. */
    private boolean traverse(Vec3d from, Vec3d to, boolean solidOnly, float minResistance) {
        // Amanatides & Woo voxel traversal, no allocation.
        int x = (int) Math.floor(from.x());
        int y = (int) Math.floor(from.y());
        int z = (int) Math.floor(from.z());
        int endX = (int) Math.floor(to.x());
        int endY = (int) Math.floor(to.y());
        int endZ = (int) Math.floor(to.z());

        double dirX = to.x() - from.x();
        double dirY = to.y() - from.y();
        double dirZ = to.z() - from.z();

        int stepX = dirX > 0 ? 1 : dirX < 0 ? -1 : 0;
        int stepY = dirY > 0 ? 1 : dirY < 0 ? -1 : 0;
        int stepZ = dirZ > 0 ? 1 : dirZ < 0 ? -1 : 0;

        double tDeltaX = stepX == 0 ? Double.MAX_VALUE : Math.abs(1.0 / dirX);
        double tDeltaY = stepY == 0 ? Double.MAX_VALUE : Math.abs(1.0 / dirY);
        double tDeltaZ = stepZ == 0 ? Double.MAX_VALUE : Math.abs(1.0 / dirZ);

        double tMaxX = stepX == 0 ? Double.MAX_VALUE
                : ((stepX > 0 ? x + 1 - from.x() : from.x() - x)) * tDeltaX;
        double tMaxY = stepY == 0 ? Double.MAX_VALUE
                : ((stepY > 0 ? y + 1 - from.y() : from.y() - y)) * tDeltaY;
        double tMaxZ = stepZ == 0 ? Double.MAX_VALUE
                : ((stepZ > 0 ? z + 1 - from.z() : from.z() - z)) * tDeltaZ;

        int guard = 0;
        while (guard++ < 512) {
            if (x == endX && y == endY && z == endZ) {
                return false;
            }
            double tMax = Math.min(tMaxX, Math.min(tMaxY, tMaxZ));
            if (tMaxX == tMax) {
                x += stepX;
                tMaxX += tDeltaX;
            } else if (tMaxY == tMax) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
            Vec3i cell = new Vec3i(x, y, z);
            if (!this.contains(cell)) {
                return false;
            }
            BlockSnapshot block = this.at(cell);
            if (solidOnly ? block.solid() : block.blastResistance() > minResistance) {
                return true;
            }
        }
        return false;
    }
}
