package com.github.luiox.fcrystalnet.poc.command;

import com.github.luiox.fcrystalnet.poc.player.FakePlayerManager;
import net.fabricmc.fabric.api.client.message.v1.ClientSendMessageEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

/**
 * The POC's chat commands, hooked at the send boundary: anything typed as
 * {@code .fcn ...} is consumed before it ever becomes a chat packet, so the commands
 * work in any world without a server-side counterpart.
 *
 * <ul>
 *   <li>{@code .fcn fakeplayer place} - spawn a fake player at your feet</li>
 *   <li>{@code .fcn fakeplayer run} - spawn one that patrols back and forth</li>
 *   <li>{@code .fcn fakeplayer reset} - remove all fake players</li>
 * </ul>
 */
public final class PocCommands {

    private static final String PREFIX = ".fcn ";

    private PocCommands() {
    }

    public static void register() {
        ClientSendMessageEvents.ALLOW_CHAT.register(message -> {
            if (!message.startsWith(PREFIX)) {
                return true;
            }
            handle(message.substring(PREFIX.length()).trim());
            return false;
        });
    }

    private static void handle(String rest) {
        switch (rest) {
            case "fakeplayer place" -> FakePlayerManager.place();
            case "fakeplayer run" -> FakePlayerManager.place(true);
            case "fakeplayer reset" -> FakePlayerManager.reset();
            default -> feedback("unknown command '.fcn " + rest + "'"
                    + " - try: fakeplayer place | run | reset");
        }
    }

    private static void feedback(String msg) {
        Minecraft mc = Minecraft.getInstance();
        if (mc.player != null) {
            mc.player.displayClientMessage(Component.literal("[fcpoc] " + msg), false);
        }
    }
}
