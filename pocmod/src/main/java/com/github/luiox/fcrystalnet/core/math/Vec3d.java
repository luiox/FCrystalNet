package com.github.luiox.fcrystalnet.core.math;

/**
 * An immutable double position, used for the quantities the game itself keeps in double
 * precision (eye position, crystal centre, linear velocity).
 *
 * <p>Same reasoning as {@link Vec3i} for not reusing a Minecraft type, plus one more:
 * {@code Vec3} declares its components {@code final}, so a record here is not just
 * portable but strictly a better fit than the alternative.
 *
 * <p>The centred form used all over crystal code is {@link #centered(double)} - a block
 * position's visual centre is {@code (x + 0.5, y, z + 0.5)}, not {@code (x, y, z)}. That
 * offset is easy to get subtly wrong and it changes damage, so it is named once here.
 */
public record Vec3d(double x, double y, double z) {

    public static final Vec3d ZERO = new Vec3d(0.0, 0.0, 0.0);

    public Vec3d offset(double dx, double dy, double dz) {
        return new Vec3d(this.x + dx, this.y + dy, this.z + dz);
    }

    public Vec3d scale(double factor) {
        return new Vec3d(this.x * factor, this.y * factor, this.z * factor);
    }

    /**
     * The centre of the block at {@code pos}: half a block in x and z, but not in y,
     * because the block's own origin is already its bottom face.
     *
     * <p>Every damage calculation in {@code crystal} needs exactly this, and getting it
     * wrong is silent - the result is merely "slightly wrong damage" rather than an error.
     */
    public static Vec3d centered(Vec3i pos) {
        return new Vec3d(pos.x() + 0.5, pos.y(), pos.z() + 0.5);
    }

    public double distanceTo(Vec3d other) {
        double dx = this.x - other.x;
        double dy = this.y - other.y;
        double dz = this.z - other.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    public double distanceSquaredTo(Vec3d other) {
        double dx = this.x - other.x;
        double dy = this.y - other.y;
        double dz = this.z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }

    public double horizontalDistanceTo(Vec3d other) {
        double dx = this.x - other.x;
        double dz = this.z - other.z;
        return Math.sqrt(dx * dx + dz * dz);
    }

    public Vec3i floor() {
        return new Vec3i((int) Math.floor(this.x), (int) Math.floor(this.y), (int) Math.floor(this.z));
    }

    @Override
    public String toString() {
        return "(" + this.x + ", " + this.y + ", " + this.z + ')';
    }
}
