package com.github.luiox.fcrystalnet.core.snapshot;

import com.github.luiox.fcrystalnet.core.math.Vec3d;
import java.util.List;

/**
 * The local player, as the decision layer may see it. Immutable.
 *
 * <p>The inventory views use the {@code InventoryMenu} slot geometry so a decision can
 * name a click target directly: armour 5-8, main inventory 9-35, hotbar 36-44, offhand
 * 45. The arrays here are the menu's backing stores in that order - {@code hotbar} stays
 * its own field because half the client reads it.
 *
 * @param pos          feet position
 * @param eyeHeight    current eye height (sneaking changes it)
 * @param hotbar       nine slots, index 0-8, menu slots 36-44
 * @param selectedSlot currently held hotbar slot, 0-8
 * @param mainInventory 27 slots, menu slots 9-35
 * @param armorSlots   four slots in menu order: head, chest, legs, feet - menu slots 5-8
 * @param offhand      the offhand stack, menu slot 45
 * @param health       current health
 * @param maxHealth    maximum health
 * @param armor        armour points
 * @param armorToughness armour toughness
 * @param protection   explosion-protection EPF, capped at 20
 * @param usingItem    whether an item is in use (shield, bow, turtle potion...) - the
 *                     flag that makes the interaction manager refuse slot switches
 * @param onGround     ground state, needed by movement packets and prediction
 * @param yaw          the player's (client) look yaw
 * @param pitch        the player's (client) look pitch
 * @param predicted    the player's own extrapolated positions, one per step; the self
 *                     side of the hurtTime asymmetry takes the worst of current and these
 */
public record SelfSnapshot(Vec3d pos, double eyeHeight,
                           ItemStackSnapshot[] hotbar, int selectedSlot,
                           ItemStackSnapshot[] mainInventory,
                           ItemStackSnapshot[] armorSlots,
                           ItemStackSnapshot offhand,
                           float health, float maxHealth,
                           float armor, float armorToughness, float protection,
                           boolean usingItem, boolean onGround,
                           float yaw, float pitch,
                           List<Vec3d> predicted) {

    /** The {@code InventoryMenu} slot of the armour array's first entry (head). */
    public static final int ARMOR_CONTAINER_BASE = 5;
    /** The {@code InventoryMenu} slot of the main-inventory array's first entry. */
    public static final int MAIN_CONTAINER_BASE = 9;
    /** The {@code InventoryMenu} slot of the hotbar array's first entry. */
    public static final int HOTBAR_CONTAINER_BASE = 36;
    /** The {@code InventoryMenu} slot of the offhand stack. */
    public static final int OFFHAND_CONTAINER_SLOT = 45;

    public SelfSnapshot {
        hotbar = normalize(hotbar);
        mainInventory = normalize(mainInventory);
        armorSlots = normalize(armorSlots);
        offhand = offhand == null ? ItemStackSnapshot.EMPTY : offhand;
        predicted = predicted == null ? List.of() : List.copyOf(predicted);
    }

    /** Clones and de-nulls a slot array: an absent stack reads as {@link ItemStackSnapshot#EMPTY}. */
    private static ItemStackSnapshot[] normalize(ItemStackSnapshot[] slots) {
        if (slots == null) {
            return new ItemStackSnapshot[0];
        }
        ItemStackSnapshot[] out = new ItemStackSnapshot[slots.length];
        for (int i = 0; i < slots.length; i++) {
            out[i] = slots[i] == null ? ItemStackSnapshot.EMPTY : slots[i];
        }
        return out;
    }

    /** A self snapshot without prediction (static player, or prediction not enabled). */
    public static SelfSnapshot of(Vec3d pos, double eyeHeight, ItemStackSnapshot[] hotbar, int selectedSlot,
                                  ItemStackSnapshot[] mainInventory, ItemStackSnapshot[] armorSlots,
                                  ItemStackSnapshot offhand,
                                  float health, float maxHealth, float armor, float armorToughness,
                                  float protection, boolean usingItem, boolean onGround,
                                  float yaw, float pitch) {
        return new SelfSnapshot(pos, eyeHeight, hotbar, selectedSlot, mainInventory, armorSlots,
                offhand, health, maxHealth, armor, armorToughness, protection, usingItem, onGround,
                yaw, pitch, List.of());
    }

    public Vec3d eyePos() {
        return new Vec3d(this.pos.x(), this.pos.y() + this.eyeHeight, this.pos.z());
    }

    /** The held stack, or {@link ItemStackSnapshot#EMPTY}. */
    public ItemStackSnapshot held() {
        if (this.selectedSlot < 0 || this.selectedSlot >= this.hotbar.length) {
            return ItemStackSnapshot.EMPTY;
        }
        return this.hotbar[this.selectedSlot];
    }

    /** The first hotbar slot holding {@code itemId}, or -1. */
    public int findSlot(String itemId) {
        for (int i = 0; i < this.hotbar.length; i++) {
            if (this.hotbar[i].is(itemId)) {
                return i;
            }
        }
        return -1;
    }

    /** The {@code InventoryMenu} slot of {@code hotbar[i]}: 36-44. */
    public int hotbarContainerSlot(int index) {
        return HOTBAR_CONTAINER_BASE + index;
    }

    /** The {@code InventoryMenu} slot of {@code mainInventory[i]}: 9-35. */
    public int mainContainerSlot(int index) {
        return MAIN_CONTAINER_BASE + index;
    }

    /** The {@code InventoryMenu} slot of {@code armorSlots[i]}: 5-8. */
    public int armorContainerSlot(int index) {
        return ARMOR_CONTAINER_BASE + index;
    }
}
