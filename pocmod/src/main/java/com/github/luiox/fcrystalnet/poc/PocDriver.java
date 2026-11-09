package com.github.luiox.fcrystalnet.poc;

import com.github.luiox.fcrystalnet.poc.act.MinimalInteraction;
import com.github.luiox.fcrystalnet.poc.decide.PocDecision;
import com.github.luiox.fcrystalnet.poc.features.FeatureBuilder;
import com.github.luiox.fcrystalnet.poc.infer.CrystalNetBridge;
import com.github.luiox.fcrystalnet.poc.perception.MinimalPerception;
import com.github.luiox.fcrystalnet.core.math.Vec3d;
import com.github.luiox.fcrystalnet.core.math.Vec3i;
import com.github.luiox.fcrystalnet.core.snapshot.SelfSnapshot;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The POC's loop: toggle, cadence, one action per tick, and the live-regret log.
 *
 * <p>Every {@value #COMPUTE_EVERY} ticks while enabled: perception frame &rarr; feature
 * tensors &rarr; one JNI inference &rarr; decision (model pick + exact oracle). The model
 * ranks the cells, the exact gates decide what is performable, and the oracle's answer
 * for the same cycle lands in the log as regret: what the perfect player would have got
 * minus what the model's pick actually yields. Breaks go first whenever a performable
 * break exists - inside vanilla's invulnerable window only the biggest hit counts, so
 * stockpiling crystals is pure waste - and two explosions stay at least the window
 * apart. The driver never turns the camera: vanilla validates range on both
 * interactions, never look direction. A slot mismatch defers the action one tick
 * (select now, act next).
 */
public final class PocDriver {

    private static final Logger LOG = LoggerFactory.getLogger(PocDriver.class);

    /** Decision cadence in ticks. */
    private static final int COMPUTE_EVERY = 5;
    /** Action pacing: a placement at most every 10 ticks. */
    private static final int PLACE_EVERY = 10;
    /**
     * Vanilla's invulnerable window in ticks: a second explosion inside it deals only
     * the difference over the first, so one crystal per window - anything closer is
     * damage override waste.
     */
    private static final int DAMAGE_WINDOW_TICKS = 10;
    /** Summary heartbeat in ticks. */
    private static final int SUMMARY_EVERY = 20;
    private static final String CRYSTAL_ITEM = "minecraft:end_crystal";

    private final KeyMapping toggleKey;
    private final MinimalPerception perception = new MinimalPerception();

    private CrystalNetBridge bridge;
    private boolean enabled;
    private int retryAt;
    private int ticks;
    private int lastPlaceTick = Integer.MIN_VALUE / 2;
    private int lastBreakTick = Integer.MIN_VALUE / 2;

    /** Deferred action: the slot was switched this tick, the act happens next tick. */
    private Pending pending;

    // Live-regret accumulation, flushed by the summary heartbeat.
    private int actions;
    private int misses;
    private double regretSum;

    private PocDriver(KeyMapping toggleKey) {
        this.toggleKey = toggleKey;
    }

    public static PocDriver create(KeyMapping toggleKey) {
        return new PocDriver(toggleKey);
    }

    /** One client tick: toggle, bookkeeping, then the enabled loop. */
    public void tick() {
        while (this.toggleKey.consumeClick()) {
            this.enabled = !this.enabled;
            feedback(this.enabled ? "POC ON (G toggles off)" : "POC OFF");
            if (!this.enabled) {
                releaseBridge();
                this.pending = null;
            } else {
                this.retryAt = 0;
            }
        }
        Minecraft minecraft = Minecraft.getInstance();
        if (minecraft.player == null || minecraft.level == null) {
            if (this.enabled) {
                this.enabled = false;
                releaseBridge();
            }
            return;
        }
        this.perception.onTick();
        if (!this.enabled) {
            return;
        }
        this.ticks++;
        runPending(minecraft);
        if (this.ticks % COMPUTE_EVERY != 0) {
            return;
        }
        if (this.bridge == null && this.ticks >= this.retryAt && !acquireBridge(minecraft)) {
            return;
        }
        computeCycle(minecraft);
        if (this.ticks % SUMMARY_EVERY == 0) {
            LOG.info("[POC] summary: ticks={} actions={} missed-opportunities={} "
                            + "mean-regret={}", this.ticks, this.actions, this.misses,
                    this.actions + this.misses == 0
                            ? "n/a" : String.format("%.2f", this.regretSum / (this.actions + this.misses)));
        }
    }

    // --- inference ---------------------------------------------------------------------------

    private boolean acquireBridge(Minecraft minecraft) {
        try {
            this.bridge = CrystalNetBridge.acquire();
            feedback("inference ready (w24v2gpu)");
            return true;
        } catch (Throwable t) {
            LOG.warn("[POC] inference unavailable", t);
            feedback("inference unavailable: " + t.getMessage()
                    + " - retrying every 100 ticks");
            this.retryAt = this.ticks + 100;
            return false;
        }
    }

    private void releaseBridge() {
        if (this.bridge != null) {
            try {
                this.bridge.close();
            } catch (Throwable t) {
                LOG.warn("[POC] bridge close failed", t);
            }
            this.bridge = null;
        }
    }

    // --- one decision cycle --------------------------------------------------------------------

    private void computeCycle(Minecraft minecraft) {
        try {
            decide(minecraft);
        } catch (Throwable t) {
            // A dying cycle is logged and survived: the next one re-reads the world.
            LOG.warn("[POC] decision cycle failed", t);
        }
    }

    private void decide(Minecraft minecraft) {
        MinimalPerception.Frame frame = this.perception.frame();
        if (frame == null) {
            rateLimitedLog("no target in range");
            return;
        }
        float[] dmg = new float[FeatureBuilder.CELLS];
        float[] valid = new float[FeatureBuilder.CELLS];
        this.bridge.infer(FeatureBuilder.buildX(frame), FeatureBuilder.buildAux(frame),
                dmg, valid);
        PocDecision.Decision decision = PocDecision.decide(frame, dmg, valid);

        if (decision.place() == null && decision.breakAction() == null) {
            // Nothing acceptable - but if the oracle saw something, that is regret too.
            if (decision.placeExact() != null) {
                this.misses++;
                this.regretSum += decision.placeExact().exactDmg();
                rateLimitedLog(String.format(
                        "no acceptable place pick; oracle saw %s (%.2f dmg)",
                        decision.placeExact().candidate().pos(),
                        decision.placeExact().exactDmg()));
            } else if (decision.breakExact() != null) {
                this.misses++;
                this.regretSum += decision.breakExact().exactDmg();
                rateLimitedLog(String.format(
                        "no acceptable break pick; oracle saw %s (%.2f dmg)",
                        decision.breakExact().cell(), decision.breakExact().exactDmg()));
            }
            return;
        }

        // Break first: an unbroken crystal blocks placement, so the stockpile-then-
        // multi-pop pattern - two explosions inside one invulnerable window, the second
        // eaten by the damage override - cannot happen.
        boolean placeSide = decision.place() != null && decision.breakAction() == null;
        double regret = decision.regret(placeSide);
        this.regretSum += Math.max(0.0, regret);

        if (placeSide) {
            if (this.ticks - this.lastPlaceTick < PLACE_EVERY) {
                return;
            }
            SelfSnapshot self = frame.self();
            int slot = self.findSlot(CRYSTAL_ITEM);
            if (slot < 0) {
                rateLimitedLog("no end crystals in the hotbar");
                return;
            }
            Vec3i cell = decision.place().candidate().pos();
            if (self.selectedSlot() != slot) {
                MinimalInteraction.selectSlot(minecraft.player, slot);
                this.pending = new PendingPlace(cell, decision.place().modelDmg(),
                        decision.place().exactDmg(), regret);
                return;
            }
            act(new PendingPlace(cell, decision.place().modelDmg(),
                    decision.place().exactDmg(), regret), minecraft);
        } else {
            if (this.ticks - this.lastBreakTick < DAMAGE_WINDOW_TICKS) {
                return;
            }
            PendingBreak action = new PendingBreak(decision.breakAction().crystal().id(),
                    decision.breakAction().cell(), decision.breakAction().modelDmg(),
                    decision.breakAction().exactDmg(), regret);
            act(action, minecraft);
        }
    }

    // --- actions -------------------------------------------------------------------------------

    private void runPending(Minecraft minecraft) {
        if (this.pending == null) {
            return;
        }
        Pending action = this.pending;
        this.pending = null;
        act(action, minecraft);
    }

    private void act(Pending action, Minecraft minecraft) {
        if (action instanceof PendingPlace place) {
            boolean done = MinimalInteraction.placeCrystal(minecraft, place.cell());
            this.lastPlaceTick = this.ticks;
            this.actions++;
            LOG.info("[POC] place score=({},{}) exact=({},{}) regret={} -> {}",
                    place.cell(), String.format("%.3f", place.modelDmg()),
                    place.cell(), String.format("%.2f", place.exactDmg()),
                    String.format("%.2f", place.regret()), done ? "ok" : "refused");
            feedback(String.format("place %d,%d,%d (score %.2f / exact %.1f)",
                    place.cell().x(), place.cell().y(), place.cell().z(),
                    place.modelDmg(), place.exactDmg()));
        } else if (action instanceof PendingBreak breakAction) {
            // Vanilla's attack reach is shorter than the dataset's 4.5 placement gate.
            Vec3d lookPos = crystalLookPos(breakAction);
            if (minecraft.player.getEyePosition().distanceToSqr(
                    new net.minecraft.world.phys.Vec3(lookPos.x(), lookPos.y(), lookPos.z()))
                    > PocDecision.BREAK_ATTACK_REACH * PocDecision.BREAK_ATTACK_REACH) {
                rateLimitedLog("break pick beyond attack reach");
                return;
            }
            boolean done = MinimalInteraction.attackEntity(minecraft, breakAction.entityId());
            if (done) {
                // Only a real explosion arms the damage window; a refused attack retries
                // at the next compute cycle.
                this.lastBreakTick = this.ticks;
                this.actions++;
            }
            LOG.info("[POC] break score=({},{}) exact=({},{}) regret={} -> {}",
                    breakAction.cell(), String.format("%.3f", breakAction.modelDmg()),
                    breakAction.cell(), String.format("%.2f", breakAction.exactDmg()),
                    String.format("%.2f", breakAction.regret()), done ? "ok" : "refused");
            feedback(String.format("break @ %d,%d,%d (score %.2f / exact %.1f)",
                    breakAction.cell().x(), breakAction.cell().y(), breakAction.cell().z(),
                    breakAction.modelDmg(), breakAction.exactDmg()));
        }
    }

    /** The break look point: the crystal entity's mid height. */
    private Vec3d crystalLookPos(PendingBreak action) {
        return new Vec3d(action.cell().x() + 0.5, action.cell().y() + 1.0,
                action.cell().z() + 0.5);
    }

    // --- feedback ------------------------------------------------------------------------------

    private String lastIdleLog;
    private int lastIdleLogTick;

    private void rateLimitedLog(String message) {
        if (!message.equals(this.lastIdleLog) || this.ticks - this.lastIdleLogTick >= 100) {
            LOG.info("[POC] {}", message);
            this.lastIdleLog = message;
            this.lastIdleLogTick = this.ticks;
        }
    }

    private static void feedback(String message) {
        Minecraft minecraft = Minecraft.getInstance();
        if (minecraft.player != null) {
            minecraft.player.displayClientMessage(Component.literal("[POC] " + message), false);
        }
    }

    private sealed interface Pending permits PendingPlace, PendingBreak {
    }

    private record PendingPlace(Vec3i cell, double modelDmg, double exactDmg,
                                double regret) implements Pending {
    }

    private record PendingBreak(int entityId, Vec3i cell, double modelDmg, double exactDmg,
                                double regret) implements Pending {
    }
}
