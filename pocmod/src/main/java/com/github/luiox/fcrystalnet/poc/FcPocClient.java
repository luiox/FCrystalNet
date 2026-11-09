package com.github.luiox.fcrystalnet.poc;

import com.github.luiox.fcrystalnet.poc.command.PocCommands;
import com.github.luiox.fcrystalnet.poc.player.FakePlayerManager;
import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.KeyMapping;
import net.minecraft.resources.Identifier;
import org.lwjgl.glfw.GLFW;

/**
 * The POC mod's whole platform surface: one keybind, one tick hook, one chat-command
 * hook. No mixins - the fabric-api callbacks cover everything a POC needs, and the
 * decision/inference layers stay in plain Java that the offline golden tests share.
 */
public final class FcPocClient implements ClientModInitializer {

    public static final String MOD_ID = "fcpoc";

    private static KeyMapping toggleKey;
    private static PocDriver driver;

    @Override
    public void onInitializeClient() {
        // Registered through the vanilla controls menu - the user binds it like any
        // vanilla key, which is the whole point of using KeyMapping here.
        toggleKey = new KeyMapping("key.fcpoc.toggle", InputConstants.Type.KEYSYM,
                GLFW.GLFW_KEY_G, new KeyMapping.Category(
                        Identifier.fromNamespaceAndPath(MOD_ID, "main")));
        KeyBindingHelper.registerKeyBinding(toggleKey);

        driver = PocDriver.create(toggleKey);
        ClientTickEvents.END_CLIENT_TICK.register(client -> driver.tick());
        ClientTickEvents.END_CLIENT_TICK.register(client -> FakePlayerManager.tick());
        PocCommands.register();
    }
}
