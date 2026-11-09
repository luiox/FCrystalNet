package com.github.luiox.fcrystalnet.poc.player;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

/**
 * The fake-player roster behind {@code .fcn fakeplayer place / reset}. Entities get
 * descending ids from 1e9 (Nami's trick) so they can never collide with real network
 * entity ids.
 */
public final class FakePlayerManager {

    private static final AtomicInteger NEXT_ID = new AtomicInteger(1_000_000_000);
    private static final List<FakePlayerEntity> FAKE = new ArrayList<>(4);

    private FakePlayerManager() {
    }

    /** Spawns one static fake player at the local player's feet, looking the same way. */
    public static void place() {
        place(false);
    }

    /**
     * Spawns one fake player; {@code runner} puppets patrol a
     * {@value FakePlayerEntity#PATROL_RANGE}-block half-line at walk speed, the axis
     * alternating per spawn so two of them cross. Moving targets exercise the
     * prediction/velocity half of the feature tensors that static ones never light up.
     */
    public static void place(boolean runner) {
        Minecraft mc = Minecraft.getInstance();
        if (mc.player == null || mc.level == null) {
            feedback("no world - join a level first");
            return;
        }
        FakePlayerEntity entity = new FakePlayerEntity(mc.player,
                "fcn_puppet_" + (FAKE.size() + 1), runner, (FAKE.size() & 1) == 0);
        entity.setId(NEXT_ID.decrementAndGet());
        entity.spawn();
        FAKE.add(entity);
        feedback((runner ? "runner" : "fake player") + " placed at " +
                String.format("%.1f %.1f %.1f", entity.getX(), entity.getY(), entity.getZ())
                + " (" + FAKE.size() + " alive)");
    }

    /** One client tick: advances the runners; a left world drops the roster. */
    public static void tick() {
        if (FAKE.isEmpty()) {
            return;
        }
        Minecraft mc = Minecraft.getInstance();
        if (mc.level == null) {
            FAKE.clear();
            return;
        }
        for (FakePlayerEntity entity : FAKE) {
            entity.tickRunner();
        }
    }

    /** Removes every fake player. */
    public static void reset() {
        if (FAKE.isEmpty()) {
            feedback("no fake players to clear");
            return;
        }
        for (FakePlayerEntity entity : FAKE) {
            entity.despawn();
        }
        int n = FAKE.size();
        FAKE.clear();
        feedback("cleared " + n + " fake player(s)");
    }

    public static int count() {
        return FAKE.size();
    }

    private static void feedback(String msg) {
        Minecraft mc = Minecraft.getInstance();
        if (mc.player != null) {
            mc.player.displayClientMessage(Component.literal("[fcpoc] " + msg), false);
        }
    }
}
