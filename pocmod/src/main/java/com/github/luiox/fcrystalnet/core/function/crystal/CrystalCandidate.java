package com.github.luiox.fcrystalnet.core.function.crystal;

import com.github.luiox.fcrystalnet.core.math.Vec3i;

/**
 * One placeable crystal position from the candidate set.
 *
 * <p>The candidate set is target-centred: base-eligible positions are the ring around the
 * target's feet box, and placement positions are the union with base positions. Each
 * candidate carries whether it needs a supporting block placed first, and which block
 * provides it - so "search depth = how many obsidian we place" reduces to this one field.
 *
 * @param pos          the block cell the crystal itself will occupy
 * @param basePos      the block below, when we must place it ({@code null} when a valid
 *                     base - obsidian or bedrock - is already there)
 * @param blocksNeeded 0 or 1 today; the field exists so multi-block bases (depth &gt; 1)
 *                     can be added without changing every consumer
 */
public record CrystalCandidate(Vec3i pos, Vec3i basePos, int blocksNeeded) {

    public static CrystalCandidate onExistingBase(Vec3i pos) {
        return new CrystalCandidate(pos, null, 0);
    }

    public static CrystalCandidate needsBase(Vec3i pos, Vec3i basePos) {
        return new CrystalCandidate(pos, basePos, 1);
    }

    public boolean needsBase() {
        return this.basePos != null;
    }
}
