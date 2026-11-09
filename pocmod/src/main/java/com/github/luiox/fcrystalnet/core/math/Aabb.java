package com.github.luiox.fcrystalnet.core.math;

import com.github.luiox.fcrystalnet.core.math.Vec3d;

/**
 * An axis-aligned box in world space - the shape every entity collision and every corner
 * sample in the damage model is expressed against.
 *
 * <p>{@code :core} has no Minecraft dependency, so this is our own; it exists because
 * {@link Vec3d} alone cannot carry an extent, and because the exposure sampling needs the
 * eight corners as data rather than arithmetic at each use site.
 */
public record Aabb(double minX, double minY, double minZ, double maxX, double maxY, double maxZ) {

    /** A box of footprint {@code width} (x/z) and height {@code height} centred on {@code center}. */
    public static Aabb around(Vec3d center, double width, double height) {
        double half = width / 2.0;
        return new Aabb(center.x() - half, center.y(), center.z() - half,
                center.x() + half, center.y() + height, center.z() + half);
    }

    public Aabb offset(double dx, double dy, double dz) {
        return new Aabb(this.minX + dx, this.minY + dy, this.minZ + dz,
                this.maxX + dx, this.maxY + dy, this.maxZ + dz);
    }

    /** Grows the box by {@code d} on every side. */
    public Aabb inflate(double d) {
        return new Aabb(this.minX - d, this.minY - d, this.minZ - d,
                this.maxX + d, this.maxY + d, this.maxZ + d);
    }

    public boolean contains(Vec3d point) {
        return point.x() >= this.minX && point.x() <= this.maxX
                && point.y() >= this.minY && point.y() <= this.maxY
                && point.z() >= this.minZ && point.z() <= this.maxZ;
    }

    public boolean intersects(Aabb other) {
        return this.minX <= other.maxX && this.maxX >= other.minX
                && this.minY <= other.maxY && this.maxY >= other.minY
                && this.minZ <= other.maxZ && this.maxZ >= other.minZ;
    }

    public Vec3d center() {
        return new Vec3d((this.minX + this.maxX) / 2.0, (this.minY + this.maxY) / 2.0,
                (this.minZ + this.maxZ) / 2.0);
    }

    /**
     * Corner {@code i} of the box, bit 0 = min/max x, bit 1 = min/max y, bit 2 = min/max z.
     * The exposure sampler iterates these.
     */
    public Vec3d corner(int i) {
        return new Vec3d(
                (i & 1) == 0 ? this.minX : this.maxX,
                (i & 2) == 0 ? this.minY : this.maxY,
                (i & 4) == 0 ? this.minZ : this.maxZ);
    }
}
