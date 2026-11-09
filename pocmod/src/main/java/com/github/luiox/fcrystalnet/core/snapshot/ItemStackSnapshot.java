package com.github.luiox.fcrystalnet.core.snapshot;

import com.github.luiox.fcrystalnet.core.math.Vec3d;

/**
 * One inventory stack: item id and count, nothing else. Item ids are strings
 * ({@code "minecraft:obsidian"}) so tests can handwrite them and the config system can
 * whitelist them without a Minecraft import.
 */
public record ItemStackSnapshot(String itemId, int count) {

    public static final ItemStackSnapshot EMPTY = new ItemStackSnapshot("", 0);

    public boolean isEmpty() {
        return this.count <= 0 || this.itemId.isEmpty();
    }

    public boolean is(String itemId) {
        return this.count > 0 && this.itemId.equals(itemId);
    }
}
