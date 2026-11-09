package com.github.luiox.fcrystalnet.poc.perception;

import com.github.luiox.fcrystalnet.core.math.Aabb;
import com.github.luiox.fcrystalnet.core.math.Vec3d;
import com.github.luiox.fcrystalnet.core.math.Vec3i;
import com.github.luiox.fcrystalnet.core.snapshot.BlockSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.EntitySnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.ItemStackSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.SelfSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.TargetSnapshot;
import com.github.luiox.fcrystalnet.core.snapshot.WorldRegionSnapshot;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.enchantment.Enchantments;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;

/**
 * The POC's minimal perception: region classify, entity snapshots, armour EPF and
 * prediction, sized to the dataset's crop conventions. Everything handed out is an immutable {@code core}-snapshot taken on the
 * client thread, so the decision layer and the offline golden tests read the same
 * shapes the teacher labelled.
 *
 * <p>Two region conventions come from the dataset: the region is 25x16x25,
 * xz +-12 around the target's foot cell, y from -6 to +9 relative to it (the exact
 * window the dataset uses), and the model crop is
 * 13x7x13 at the target's foot cell + (-6, -3, -6).
 */
public final class MinimalPerception {

    /** How far from the player an entity is still interesting, in blocks. */
    private static final double ENTITY_RANGE = 16.0;
    /** Target-selection range: the fight's definition, same as CandidateStage.BREAK_RANGE. */
    private static final double TARGET_RANGE = 12.0;

    /** Region half-width in xz; the y window spans -6..+9 relative to the target's feet. */
    public static final int REGION_RADIUS_XZ = 12;
    public static final int REGION_MIN_DY = -6;
    public static final int REGION_SIZE_X = 2 * REGION_RADIUS_XZ + 1; // 25
    public static final int REGION_SIZE_Y = 16;
    public static final int REGION_SIZE_Z = 2 * REGION_RADIUS_XZ + 1; // 25

    /** Prediction covers two ticks, sampled at half-tick steps.
     *  Public because the aux vector encodes the step count (features.py's aim_steps). */
    public static final int PREDICTION_STEPS = 4;
    public static final double STEP_TICKS = 0.5;

    /** Feet positions from the previous tick, for the frame-delta extrapolation. */
    private final Map<Integer, Vec3d> lastPositions = new HashMap<>();
    private final Map<Integer, Vec3d> velocities = new HashMap<>();

    /** Called once per client tick before any cycle reads perception. */
    public void onTick() {
        ClientLevel level = Minecraft.getInstance().level;
        if (level == null) {
            this.lastPositions.clear();
            this.velocities.clear();
            return;
        }
        Map<Integer, Vec3d> positions = new HashMap<>();
        Map<Integer, Vec3d> velocities = new HashMap<>();
        for (Entity entity : level.entitiesForRendering()) {
            net.minecraft.world.phys.Vec3 pos = entity.position();
            Vec3d current = new Vec3d(pos.x, pos.y, pos.z);
            positions.put(entity.getId(), current);
            Vec3d last = this.lastPositions.get(entity.getId());
            velocities.put(entity.getId(),
                    last == null ? Vec3d.ZERO : new Vec3d(current.x() - last.x(),
                            current.y() - last.y(), current.z() - last.z()));
        }
        this.lastPositions.clear();
        this.lastPositions.putAll(positions);
        this.velocities.clear();
        this.velocities.putAll(velocities);
    }

    /**
     * One perception cycle: self, target, entities, region. {@code target == null} is
     * the IDLE gate - the driver switches off for the cycle.
     */
    public Frame frame() {
        SelfSnapshot self = self();
        if (self == null) {
            return null;
        }
        List<EntitySnapshot> entities = entities();
        EntitySnapshot chosen = target(self, entities);
        if (chosen == null) {
            return null;
        }
        TargetSnapshot target = new TargetSnapshot(chosen, predict(chosen.id(), chosen.pos()),
                true);
        return new Frame(self, target, entities,
                snapshotRegionAround(target.current().pos()),
                PREDICTION_STEPS, STEP_TICKS);
    }

    /**
     * Everything one decision cycle consumes, built together so it agrees. The two
     * prediction-convention fields ride along because the aux vector and the velocity
     * recovery are defined against them: {@code aimSteps} is the prediction list's
     * length and {@code stepTicks} the tick span per step - the dataset predicts at integer
     * ticks (span {@code aimSteps} ticks), the live perception at half ticks, and both must recover the same per-tick velocity.
     */
    public record Frame(SelfSnapshot self, TargetSnapshot target,
                        List<EntitySnapshot> entities, WorldRegionSnapshot region,
                        int aimSteps, double stepTicks) {
    }

    // --- region ---------------------------------------------------------------------------

    /** The 25x16x25 target-centred region (the dataset's window). No cache - 10k reads per
     *  cycle at 5-tick cadence is nothing, and staleness bugs are worth less than this. */
    private WorldRegionSnapshot snapshotRegionAround(Vec3d center) {
        Vec3i cell = center.floor();
        Vec3i min = new Vec3i(cell.x() - REGION_RADIUS_XZ, cell.y() + REGION_MIN_DY,
                cell.z() - REGION_RADIUS_XZ);
        ClientLevel level = Minecraft.getInstance().level;
        BlockSnapshot[] blocks = new BlockSnapshot[REGION_SIZE_X * REGION_SIZE_Y * REGION_SIZE_Z];
        int index = 0;
        for (int y = 0; y < REGION_SIZE_Y; y++) {
            for (int z = 0; z < REGION_SIZE_Z; z++) {
                for (int x = 0; x < REGION_SIZE_X; x++) {
                    blocks[index++] = level == null ? BlockSnapshot.AIR
                            : classify(level.getBlockState(new BlockPos(
                                    min.x() + x, min.y() + y, min.z() + z)));
                }
            }
        }
        return WorldRegionSnapshot.of(min, REGION_SIZE_X, REGION_SIZE_Y, REGION_SIZE_Z,
                blocks, Map.of(), difficultyOf(level));
    }

    private static WorldRegionSnapshot.Difficulty difficultyOf(ClientLevel level) {
        if (level == null) {
            return WorldRegionSnapshot.Difficulty.NORMAL;
        }
        return switch (level.getDifficulty()) {
            case PEACEFUL -> WorldRegionSnapshot.Difficulty.PEACEFUL;
            case EASY -> WorldRegionSnapshot.Difficulty.EASY;
            case HARD -> WorldRegionSnapshot.Difficulty.HARD;
            default -> WorldRegionSnapshot.Difficulty.NORMAL;
        };
    }

    /** Maps a live block state onto the decision layer's coarse classes. */
    private static BlockSnapshot classify(BlockState state) {
        if (state.isAir()) {
            return BlockSnapshot.AIR;
        }
        if (state.canBeReplaced()) {
            return BlockSnapshot.REPLACEABLE;
        }
        if (state.is(Blocks.OBSIDIAN)) {
            return BlockSnapshot.OBSIDIAN;
        }
        if (state.is(Blocks.BEDROCK)) {
            return BlockSnapshot.BEDROCK;
        }
        return state.getBlock().getExplosionResistance() >= 6.0f
                ? BlockSnapshot.STONE
                : BlockSnapshot.OTHER;
    }

    // --- entities -------------------------------------------------------------------------

    /** Every interesting entity except self; the target and the crystals stay in. */
    public List<EntitySnapshot> entities() {
        List<EntitySnapshot> out = new ArrayList<>(8);
        ClientLevel level = Minecraft.getInstance().level;
        LocalPlayer self = Minecraft.getInstance().player;
        if (level == null || self == null) {
            return out;
        }
        for (Entity entity : level.entitiesForRendering()) {
            if (entity == self) {
                continue;
            }
            if (entity.position().distanceTo(self.position()) > ENTITY_RANGE) {
                continue;
            }
            out.add(snapshot(entity));
        }
        return out;
    }

    /** Nearest alive living entity (a fake player qualifies) within {@link #TARGET_RANGE}. */
    private static EntitySnapshot target(SelfSnapshot self, List<EntitySnapshot> entities) {
        EntitySnapshot best = null;
        double bestDist = Double.MAX_VALUE;
        for (EntitySnapshot entity : entities) {
            // Crystals are fight participants, but never the target - they are the
            // break side's subjects.
            if (entity.type().equals("minecraft:end_crystal")) {
                continue;
            }
            if (entity.dead() || entity.health() <= 0.0f) {
                continue;
            }
            double dist = entity.pos().distanceTo(self.pos());
            if (dist <= TARGET_RANGE && dist < bestDist) {
                best = entity;
                bestDist = dist;
            }
        }
        return best;
    }

    private EntitySnapshot snapshot(Entity entity) {
        net.minecraft.world.phys.Vec3 pos = entity.position();
        Vec3d feet = new Vec3d(pos.x, pos.y, pos.z);
        if (!(entity instanceof LivingEntity living)) {
            // End crystals and similar have no health semantics worth reading.
            return new EntitySnapshot(entity.getId(), typeOf(entity), feet,
                    Aabb.around(feet, entity.getBbWidth(), entity.getBbHeight()),
                    1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0, !entity.isAlive(), List.of());
        }
        return new EntitySnapshot(entity.getId(), typeOf(entity), feet,
                Aabb.around(feet, entity.getBbWidth(), entity.getBbHeight()),
                living.getHealth(), living.getMaxHealth(),
                living.getArmorValue(), (float) living.getAttributeValue(Attributes.ARMOR_TOUGHNESS),
                protectionOf(living), resistanceLevel(living),
                living.hurtTime, !living.isAlive(), predict(living.getId(), feet));
    }

    private static String typeOf(Entity entity) {
        return net.minecraft.world.entity.EntityType.getKey(entity.getType()).toString();
    }

    /** Constant-velocity extrapolation shared by target and self. */
    private List<Vec3d> predict(int entityId, Vec3d current) {
        Vec3d velocity = velocityOf(entityId);
        if (velocity == null || (velocity.x() == 0.0 && velocity.y() == 0.0
                && velocity.z() == 0.0)) {
            return List.of();
        }
        Vec3d[] out = new Vec3d[PREDICTION_STEPS];
        for (int i = 0; i < PREDICTION_STEPS; i++) {
            double t = STEP_TICKS * (i + 1);
            out[i] = current.offset(velocity.x() * t, velocity.y() * t, velocity.z() * t);
        }
        return List.of(out);
    }

    private Vec3d velocityOf(int entityId) {
        return this.velocities.get(entityId);
    }

    // --- self -----------------------------------------------------------------------------

    public SelfSnapshot self() {
        LocalPlayer player = Minecraft.getInstance().player;
        if (player == null) {
            return null;
        }
        var nonEquipment = player.getInventory().getNonEquipmentItems();
        ItemStackSnapshot[] hotbar = new ItemStackSnapshot[9];
        for (int i = 0; i < 9; i++) {
            hotbar[i] = stack(nonEquipment.get(i));
        }
        ItemStackSnapshot[] mainInventory = new ItemStackSnapshot[27];
        for (int i = 0; i < 27; i++) {
            mainInventory[i] = stack(nonEquipment.get(9 + i));
        }
        ItemStackSnapshot[] armorSlots = new ItemStackSnapshot[]{
                stack(player.getItemBySlot(EquipmentSlot.HEAD)),
                stack(player.getItemBySlot(EquipmentSlot.CHEST)),
                stack(player.getItemBySlot(EquipmentSlot.LEGS)),
                stack(player.getItemBySlot(EquipmentSlot.FEET))};
        net.minecraft.world.phys.Vec3 pos = player.position();
        Vec3d feet = new Vec3d(pos.x, pos.y, pos.z);
        return new SelfSnapshot(feet, player.getEyeHeight(), hotbar,
                player.getInventory().getSelectedSlot(),
                mainInventory, armorSlots,
                stack(player.getItemBySlot(EquipmentSlot.OFFHAND)),
                player.getHealth(), player.getMaxHealth(),
                player.getArmorValue(), (float) player.getAttributeValue(Attributes.ARMOR_TOUGHNESS),
                protectionOf(player), player.isUsingItem(), player.onGround(),
                player.getYRot(), player.getXRot(), predict(player.getId(), feet));
    }

    // --- armour classification ------------------------------------

    /** Explosion-protection EPF over the four armour slots: Protection 1/level, Blast 2/level, capped at 20. */
    private static int protectionOf(LivingEntity living) {
        int total = 0;
        for (EquipmentSlot slot : new EquipmentSlot[]{
                EquipmentSlot.HEAD, EquipmentSlot.CHEST, EquipmentSlot.LEGS, EquipmentSlot.FEET}) {
            total += epf(living.getItemBySlot(slot));
        }
        return Math.min(20, total);
    }

    private static int epf(ItemStack stack) {
        int total = 0;
        for (var entry : stack.getEnchantments().entrySet()) {
            int level = entry.getIntValue();
            if (entry.getKey().is(Enchantments.PROTECTION)) {
                total += level;
            } else if (entry.getKey().is(Enchantments.BLAST_PROTECTION)) {
                total += 2 * level;
            }
        }
        return total;
    }

    private static float resistanceLevel(LivingEntity living) {
        MobEffectInstance effect = living.getEffect(MobEffects.RESISTANCE);
        return effect == null ? 0.0f : effect.getAmplifier() + 1.0f;
    }

    private static ItemStackSnapshot stack(ItemStack stack) {
        if (stack == null || stack.isEmpty()) {
            return ItemStackSnapshot.EMPTY;
        }
        return new ItemStackSnapshot(BuiltInRegistries.ITEM.getKey(stack.getItem()).toString(),
                stack.getCount());
    }
}
