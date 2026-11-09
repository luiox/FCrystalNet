package com.github.luiox.fcrystalnet.poc.act;

import com.github.luiox.fcrystalnet.core.math.Vec3d;
import com.github.luiox.fcrystalnet.core.math.Vec3i;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.MultiPlayerGameMode;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.protocol.game.ServerboundSetCarriedItemPacket;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The POC's whole interaction surface: slot select, client look, one crystal placement,
 * one entity attack. The driver owns policy, this owns the Minecraft idiom for each step.
 *
 * <p>Placement clicks the top face of the block under the cell - the click the item's own
 * rule resolves from - and the click is built, not cast: a camera raycast would land on
 * whichever block is nearest and resolve to a neighbouring cell the decision never vetted.
 * No rotation anywhere: vanilla validates range on both interactions, never look
 * direction, so the driver runs rotation-free and the camera stays the user's.
 * The slot switch sends the carried-item packet in the same tick.
 */
public final class MinimalInteraction {

    private static final Logger LOG = LoggerFactory.getLogger(MinimalInteraction.class);

    /** How far a placement raycast looks, in blocks. */
    private static final double REACH = 4.5;

    private MinimalInteraction() {
    }

    /**
     * Selects a hotbar slot; the local write is synced with the carried-item packet
     * immediately, because vanilla's own sync runs in LocalPlayer.tick - earlier than the
     * driver's post-tick dispatch - and a same-tick useItemOn would otherwise reach the
     * server with the old slot.
     */
    public static boolean selectSlot(LocalPlayer player, int slot) {
        if (slot < 0 || slot > 8) {
            return false;
        }
        if (player.getInventory().getSelectedSlot() != slot) {
            player.getInventory().setSelectedSlot(slot);
            player.connection.send(new ServerboundSetCarriedItemPacket(slot));
        }
        return true;
    }

    /** Turns the client look at {@code target}. Unused by the driver (rotation-free
     *  mode); kept as a debugging aid. */
    public static void lookAt(LocalPlayer player, Vec3d target) {
        net.minecraft.world.phys.Vec3 eyePos = player.getEyePosition();
        double dx = target.x() - eyePos.x;
        double dy = target.y() - eyePos.y;
        double dz = target.z() - eyePos.z;
        double horizontal = Math.sqrt(dx * dx + dz * dz);
        float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0);
        float pitch = (float) -Math.toDegrees(Math.atan2(dy, horizontal));
        player.setYRot(yaw);
        player.setXRot(pitch);
        player.yRotO = yaw;
        player.xRotO = pitch;
    }

    /**
     * Places an end crystal so it occupies {@code cell}: right-clicks the top face of
     * {@code cell.below()}. Returns false when the caller is out of reach or the item
     * refuses - a refused placement is an outcome, not an exception.
     */
    public static boolean placeCrystal(Minecraft minecraft, Vec3i cell) {
        LocalPlayer player = minecraft.player;
        MultiPlayerGameMode gameMode = minecraft.gameMode;
        if (player == null || minecraft.level == null || gameMode == null) {
            return false;
        }
        BlockPos blockPos = new BlockPos(cell.x(), cell.y(), cell.z());
        Vec3 centre = new Vec3(blockPos.getX() + 0.5, blockPos.getY() + 0.5,
                blockPos.getZ() + 0.5);
        // The decision's reach gate and the server both measure from the eye - keeping
        // the same metric here so a cell that passed one gate cannot fail another.
        if (player.getEyePosition().distanceToSqr(centre) > REACH * REACH) {
            return false;
        }
        BlockHitResult blockHit = new BlockHitResult(
                new Vec3(blockPos.getX() + 0.5, blockPos.getY(), blockPos.getZ() + 0.5),
                Direction.UP, blockPos.below(), false);
        InteractionResult result = gameMode.useItemOn(player, InteractionHand.MAIN_HAND,
                blockHit);
        if (result.consumesAction()) {
            player.swing(InteractionHand.MAIN_HAND);
            return true;
        }
        LOG.info("[POC] placement at {} refused: useItemOn returned {} (hand {})",
                blockPos, result, player.getItemInHand(InteractionHand.MAIN_HAND));
        return false;
    }

    /** Attacks the entity - breaking an end crystal is an entity attack. */
    public static boolean attackEntity(Minecraft minecraft, int entityId) {
        LocalPlayer player = minecraft.player;
        MultiPlayerGameMode gameMode = minecraft.gameMode;
        if (player == null || minecraft.level == null || gameMode == null || entityId < 0) {
            return false;
        }
        Entity target = minecraft.level.getEntity(entityId);
        if (target == null) {
            return false;
        }
        gameMode.attack(player, target);
        player.swing(InteractionHand.MAIN_HAND);
        return true;
    }
}
