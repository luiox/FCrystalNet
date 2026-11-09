package com.github.luiox.fcrystalnet.core.math;

/**
 * An immutable integer position.
 *
 * <p>Deliberately not {@code net.minecraft.core.Vec3i}. Three reasons, in order of how
 * much they cost if ignored:
 *
 * <ol>
 *   <li><strong>Minecraft's own fields are private and named by Mojang.</strong> Gson
 *       serialises by field name, so using the Minecraft type would make the config file's
 *       keys depend on an upstream naming decision we do not control. A rename upstream
 *       would silently invalidate every user's config.</li>
 *   <li><strong>:core must not import {@code net.minecraft}</strong>, or the portable
 *       layer stops being portable and nothing in it can be unit tested without the game.</li>
 *   <li><strong>The Minecraft type is mutable and shared.</strong> {@code BlockPos.ZERO}
 *       is a shared constant; holding one as configuration state is shared mutable state,
 *       whose failure mode is intermittent and unreproducible.</li>
 * </ol>
 *
 * <p>Converting at the boundary costs three lines each way and buys headless-testable
 * integer maths - collision tests, range clipping, AABB overlap - with no game running.
 */
public record Vec3i(int x, int y, int z) {

    public static final Vec3i ZERO = new Vec3i(0, 0, 0);

    /** {@code BlockPos.relative} equivalent: offsets this position by a signed delta. */
    public Vec3i offset(int dx, int dy, int dz) {
        return new Vec3i(this.x + dx, this.y + dy, this.z + dz);
    }

    public Vec3i offset(int dx, int dy, int dz, int times) {
        return offset(dx * times, dy * times, dz * times);
    }

    public Vec3i down() {
        return offset(0, -1, 0);
    }

    public Vec3i up() {
        return offset(0, 1, 0);
    }

    public Vec3i below() {
        return down();
    }

    public Vec3i above() {
        return up();
    }

    /** Chebyshev distance: the number of block moves on a grid ignoring diagonals. */
    public int chebyshevDistance(Vec3i other) {
        return Math.max(Math.max(Math.abs(this.x - other.x), Math.abs(this.y - other.y)),
                Math.abs(this.z - other.z));
    }

    public int manhattanDistance(Vec3i other) {
        return Math.abs(this.x - other.x) + Math.abs(this.y - other.y) + Math.abs(this.z - other.z);
    }

    /** Squared euclidean distance, for callers that only need to compare. */
    public long distanceSquared(Vec3i other) {
        long dx = (long) this.x - other.x;
        long dy = (long) this.y - other.y;
        long dz = (long) this.z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }

    public boolean isWithin(Vec3i min, Vec3i max) {
        return this.x >= min.x && this.x <= max.x
                && this.y >= min.y && this.y <= max.y
                && this.z >= min.z && this.z <= max.z;
    }

    @Override
    public String toString() {
        return "(" + this.x + ", " + this.y + ", " + this.z + ')';
    }
}
