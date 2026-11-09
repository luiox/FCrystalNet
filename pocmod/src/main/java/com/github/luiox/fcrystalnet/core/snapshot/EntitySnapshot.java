package com.github.luiox.fcrystalnet.core.snapshot;

import com.github.luiox.fcrystalnet.core.math.Aabb;
import com.github.luiox.fcrystalnet.core.math.Vec3d;
import java.util.List;

/**
 * One entity, as the decision layer is allowed to see it. Immutable; the fabric side
 * re-reads it from the world on the client thread and nothing here ever points back at a
 * live object.
 *
 * @param id              the entity's numeric id - the handle the interaction layer
 *                        attacks through
 * @param type            entity type id, e.g. {@code "minecraft:end_crystal"}
 * @param pos             position (feet centre for living entities, as the game defines it)
 * @param box             collision box
 * @param health          current health
 * @param maxHealth       maximum health
 * @param armor           armour points (0..20)
 * @param armorToughness  armour toughness (0..20)
 * @param protection      total explosion-protection EPF already capped at 20
 * @param resistanceLevel the Resistance status-effect level (0 = none, 4 = Resistance IV)
 * @param hurtTime        remaining hurt invulnerability ticks
 * @param dead            whether the entity is dead or removed
 * @param predicted       future positions, one per extrapolation step; empty when
 *                        prediction is not enabled for this entity
 */
public record EntitySnapshot(int id, String type, Vec3d pos, Aabb box,
                             float health, float maxHealth,
                             float armor, float armorToughness, float protection,
                             float resistanceLevel,
                             int hurtTime, boolean dead,
                             List<Vec3d> predicted) {

    public EntitySnapshot {
        predicted = predicted == null ? List.of() : List.copyOf(predicted);
    }

    /** A snapshot without prediction - the common case for entities nobody targets. */
    public static EntitySnapshot unpredicted(int id, String type, Vec3d pos, Aabb box,
                                             float health, float maxHealth,
                                             float armor, float armorToughness, float protection,
                                             float resistanceLevel,
                                             int hurtTime, boolean dead) {
        return new EntitySnapshot(id, type, pos, box, health, maxHealth, armor, armorToughness,
                protection, resistanceLevel, hurtTime, dead, List.of());
    }

    public boolean isTargetable() {
        return !this.dead && this.health > 0.0f;
    }
}
