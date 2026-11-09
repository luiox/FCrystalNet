package com.github.luiox.fcrystalnet.poc.player;

import com.mojang.authlib.GameProfile;
import java.util.UUID;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.PlayerInfo;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.client.player.RemotePlayer;
import net.minecraft.world.entity.Entity.RemovalReason;

/**
 * A client-side-only stand-in player, ported from the Meteor/Nami fake-player pattern
 * (Nami runs this exact 1.21.11 mojmap stack): {@code extends RemotePlayer}, constructed
 * from a bare {@code (level, GameProfile)} - no Connection stub - spawned with
 * {@code level.addEntity}, despawned with {@code removeEntity(DISCARDED)}.
 *
 * <p>The server never hears about this entity, so crystal explosions will not damage it;
 * it is a target dummy for the decision pipeline. Everything client-side sees it as a
 * real player: perception enumerates it, its box blocks crystal placements, the attack
 * path resolves it through {@code level.getEntity(id)}.
 */
public final class FakePlayerEntity extends RemotePlayer {

    /** Runner patrol half-length, in blocks (a 6-block lap). */
    public static final double PATROL_RANGE = 3.0;
    /** Vanilla walk speed, blocks per tick (~4.3 m/s). */
    public static final double WALK_SPEED = 0.216;

    /** Runner puppets walk the anchor line; static ones stay put. */
    private final boolean runner;
    /** The patrol axis: true = east-west (x), false = north-south (z). */
    private final boolean patrolX;
    private final double anchorX;
    private final double anchorZ;
    private double waypoint;

    FakePlayerEntity(LocalPlayer self, String name, boolean runner, boolean patrolX) {
        super(Minecraft.getInstance().level, new GameProfile(UUID.randomUUID(), name));
        copyPosition(self);

        yRotO = getYRot();
        xRotO = getXRot();
        yHeadRot = self.yHeadRot;
        yHeadRotO = yHeadRot;
        yBodyRot = self.yBodyRot;
        yBodyRotO = yBodyRot;

        this.runner = runner;
        this.patrolX = patrolX;
        this.anchorX = getX();
        this.anchorZ = getZ();
        this.waypoint = (patrolX ? this.anchorX : this.anchorZ) + PATROL_RANGE;
        if (runner) {
            setSprinting(true);
        }

        // A faithful dummy: same attribute roll-ups (armour, toughness, speed) and the
        // same pose, so perception's box/attribute reads match a real stand-in.
        getAttributes().assignAllValues(self.getAttributes());
        setPose(self.getPose());
        setHealth(self.getHealth());
    }

    /** One client tick of the patrol; static puppets return immediately. */
    void tickRunner() {
        if (!this.runner || this.isRemoved()) {
            return;
        }
        double along = this.patrolX ? getX() : getZ();
        double delta = this.waypoint - along;
        if (Math.abs(delta) <= WALK_SPEED) {
            // Reached the end: turn around at the opposite end of the lap.
            this.waypoint = along + (this.waypoint > (this.patrolX ? this.anchorX : this.anchorZ)
                    ? -PATROL_RANGE : PATROL_RANGE);
            delta = this.waypoint - along;
        }
        double step = Math.signum(delta) * WALK_SPEED;
        setPos(this.patrolX ? getX() + step : getX(), getY(),
                this.patrolX ? getZ() : getZ() + step);
        // Vanilla yaw: 0 = +z, -90 = +x. Facing the walk keeps the dummy honest.
        float yaw = this.patrolX ? (step > 0 ? -90.0f : 90.0f)
                : (step > 0 ? 0.0f : 180.0f);
        setYRot(yaw);
        yHeadRot = yaw;
        yBodyRot = yaw;
        yRotO = yaw;
        yHeadRotO = yaw;
        yBodyRotO = yaw;
    }

    public void spawn() {
        unsetRemoved();
        Minecraft.getInstance().level.addEntity(this);
    }

    public void despawn() {
        Minecraft.getInstance().level.removeEntity(getId(), RemovalReason.DISCARDED);
        setRemoved(RemovalReason.DISCARDED);
    }

    /**
     * The fake profile has no PlayerInfo entry (nobody ever logged in with it), so the
     * skin lookup borrows the local player's - the dummy renders wearing the operator's
     * skin instead of the default Steve. Same fallback Meteor uses, minus their accessor
     * mixin.
     */
    @Override
    protected PlayerInfo getPlayerInfo() {
        PlayerInfo own = super.getPlayerInfo();
        if (own != null) {
            return own;
        }
        var mc = Minecraft.getInstance();
        return mc.getConnection() == null || mc.player == null
                ? null
                : mc.getConnection().getPlayerInfo(mc.player.getUUID());
    }
}
