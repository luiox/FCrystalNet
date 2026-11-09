package com.github.luiox.fcrystalnet.core.function.crystal;

import com.github.luiox.fcrystalnet.core.math.Vec3d;
import com.github.luiox.fcrystalnet.core.math.Aabb;
import com.github.luiox.fcrystalnet.core.snapshot.EntitySnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.WorldRegionSnapshot;

/**
 * The end-crystal explosion damage formula, computed against a
 * {@link WorldRegionSnapshot} - one implementation, shared by everything, exactly so the
 * "two copies drift" failure mode cannot exist.
 *
 * <p><strong>Vanilla-exact, pinned against the 1.21.11 bytecode</strong>
 * ({@code ServerExplosion} + {@code ExplosionDamageCalculator#getEntityDamageAmount}):
 *
 * <pre>
 * doubleSize = radius * 2                              radius = 6, so 12
 * distance   = |entity feet - explosion centre|        feet, not box centre
 * impact     = (1 - distance / doubleSize) * exposure  exposure = vanilla grid sampling
 * damage     = (impact*impact + impact) / 2 * 7 * doubleSize + 1     (no floor - 1.21.11
 *              returns the raw double; the pre-1.17 floor is gone)
 * then reductions, in order: difficulty, armour+toughness, Resistance, protection EPF
 * </pre>
 *
 * <p>Conventions the callers must respect:
 *
 * <ul>
 *   <li><strong>The explosion centre is the crystal cell's bottom-centre</strong>
 *       ({@code x+0.5, y, z+0.5}) - that is the point vanilla spawns the explosion at
 *       ({@code EndCrystalItem.useOn} clicks the support block's top face). Callers pass
 *       {@code Vec3d.centered(cell)}; adding a Y offset here would double-count.</li>
 *   <li><strong>Exposure is vanilla's grid sampling</strong> over the entity box: step
 *       {@code 1/(2*diff+1)} per axis, ray from each grid point to the explosion centre
 *       through the snapshot with {@code COLLIDER} semantics - any block with a collision
 *       shape occludes (verified in the bytecode), fluids do not. Twenty rays for a
 *       standing player. There is no side-mask approximation any more: it disagreed with
 *       vanilla by whole hearts behind partial cover, which is exactly the kind of wrong
 *       a damage model must not be.</li>
 *   <li><strong>The difficulty scales before the reductions</strong> (EASY:
 *       {@code min(d/2+1, d)}, HARD: {@code &times;1.5}), and Resistance follows vanilla's
 *       16% per level, capped at 80% - not meteor's uncapped 20% per level.</li>
 *   <li><strong>Region edges are the caller's responsibility.</strong> Reads outside the
 *       snapshot return AIR and rays that leave the region end unobstructed - so a region
 *       smaller than {@code radius 6 + entity box} quietly over-estimates exposure (walls
 *       beyond the edge stop blocking). The fabric perception sizes the region from the
 *       modules' declared needs (crystal: {@code CandidateStage.RADIUS + 2}), which keeps
 *       every ray inside real geometry; the model itself cannot detect the shortfall.</li>
 * </ul>
 *
 * <p>Pure: no Minecraft type, no global state. The overlay inside the region - the
 * hypothetical obsidian - participates in occlusion like any real block.
 */
public final class DamageModel {

    /** Crystal explosion radius. The vanilla divisor and multiplier are both {@code 2 * radius}. */
    public static final double POWER = 6.0;
    /** The vanilla divisor and multiplier: {@code radius * 2}. Point-blank, unobstructed, naked: 85. */
    public static final double DOUBLE_SIZE = 2.0 * POWER;

    private DamageModel() {
    }

    /**
     * Damage of an explosion at {@code source} against {@code target}'s box evaluated at
     * {@code aimPos} (its predicted position, or its current one). The distance is
     * measured from the entity's feet, exactly as {@code Entity#distanceToSqr} measures.
     *
     * @param region    the block region, overlay and difficulty included
     * @param source    the explosion centre - for a crystal, the cell's bottom-centre
     * @param target    the entity
     * @param aimPos    where the entity's box is anchored for this evaluation
     * @return damage after reductions, in half-hearts
     */
    public static double explosionDamage(WorldRegionSnapshot region, Vec3d source,
                                         EntitySnapshot target, Vec3d aimPos) {
        Aabb box = target.box().offset(aimPos.x() - target.pos().x(),
                aimPos.y() - target.pos().y(), aimPos.z() - target.pos().z());
        // Vanilla measures from the entity's position() - the feet, the box's bottom
        // centre - not from the box centre. A standing player's box centre sits 0.9 up;
        // using it under-charged every crystal by more than the placement thresholds.
        double distance = source.distanceTo(new Vec3d(box.minX() + (box.maxX() - box.minX()) / 2.0,
                box.minY(), box.minZ() + (box.maxZ() - box.minZ()) / 2.0));
        double distancedSize = distance / DOUBLE_SIZE;
        // Vanilla's gate: beyond twice the radius the entity is untouched - no exposure
        // sampling, no minimum damage.
        if (distancedSize > 1.0) {
            return 0.0;
        }
        double exposure = exposure(region, source, box);
        double impact = Math.clamp((1.0 - distancedSize) * exposure, 0.0, 1.0);
        double base = (impact * impact + impact) / 2.0 * 7.0 * DOUBLE_SIZE + 1.0;
        return reduce(base, target, region.difficulty());
    }

    /**
     * The visible fraction of {@code box} from {@code source}: vanilla's
     * {@code getSeenPercent} grid. The steps are {@code 1/(2*diff+1)} per axis, offset so
     * the grid is centred, and each grid point sends one ray to the source through the
     * snapshot's voxel traversal with collision-shape occlusion. The traversal is
     * allocation-free; the loop keeps doubles in locals, no Vec3 churn per ray.
     */
    static double exposure(WorldRegionSnapshot region, Vec3d source, Aabb box) {
        double xDiff = box.maxX() - box.minX();
        double yDiff = box.maxY() - box.minY();
        double zDiff = box.maxZ() - box.minZ();

        double xStep = 1.0 / (xDiff * 2.0 + 1.0);
        double yStep = 1.0 / (yDiff * 2.0 + 1.0);
        double zStep = 1.0 / (zDiff * 2.0 + 1.0);
        if (!(xStep > 0.0) || !(yStep > 0.0) || !(zStep > 0.0)) {
            return 0.0;
        }

        // The grid is centred on X and Z: the leftover of the whole-number sample count
        // is split. Vanilla offsets those two axes only (two Math.floor calls in the
        // bytecode); Y starts at the box floor.
        double xOffset = (1.0 - Math.floor(1.0 / xStep) * xStep) * 0.5;
        double zOffset = (1.0 - Math.floor(1.0 / zStep) * zStep) * 0.5;

        int misses = 0;
        int hits = 0;
        for (double x = box.minX() + xOffset; x <= box.maxX() + xOffset; x += xStep * xDiff) {
            for (double y = box.minY(); y <= box.maxY(); y += yStep * yDiff) {
                for (double z = box.minZ() + zOffset; z <= box.maxZ() + zOffset; z += zStep * zDiff) {
                    hits++;
                    if (!region.occludedByCollision(new Vec3d(x, y, z), source)) {
                        misses++;
                    }
                }
            }
        }
        return hits == 0 ? 0.0 : (double) misses / (double) hits;
    }

    /**
     * The vanilla reduction chain, in order, on the raw base: difficulty scaling, armour
     * + toughness, Resistance (16% per level, capped at 80% - {@code CombatRules} caps
     * the effect at level five and vanilla never exceeds that), then the protection EPF
     * (itself capped at 20 by the perception layer, i.e. 80% again).
     */
    static double reduce(double damage, EntitySnapshot target, WorldRegionSnapshot.Difficulty difficulty) {
        double d = damage;
        // Explosion damage sources scale with difficulty before every reduction.
        d = switch (difficulty) {
            case EASY -> Math.min(d / 2.0 + 1.0, d);
            case HARD -> d * 1.5;
            default -> d;
        };
        // Armour + toughness.
        double armorValue = target.armor();
        if (armorValue > 0) {
            double toughness = target.armorToughness();
            double reduction = Math.max(armorValue / 5.0, armorValue - d / (2.0 + toughness / 4.0));
            reduction = Math.min(20.0, reduction);
            d *= 1.0 - reduction / 25.0;
        }
        // Resistance status effect: vanilla applies (amplifier+1) * 4 points, clamped to
        // 20, out of 25 - 16% per level, never more than 80% total. The snapshot stores
        // amplifier+1 as the level.
        if (target.resistanceLevel() > 0) {
            double points = Math.min(target.resistanceLevel() * 4.0, 20.0);
            d *= 1.0 - points / 25.0;
        }
        // Protection + Blast Protection EPF, already capped at 20 by the perception layer.
        double epf = target.protection();
        if (epf > 0) {
            d *= 1.0 - Math.min(20.0, epf) / 25.0;
        }
        return Math.max(0.0, d);
    }
}
