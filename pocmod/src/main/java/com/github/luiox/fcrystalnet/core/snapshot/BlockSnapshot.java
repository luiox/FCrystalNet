package com.github.luiox.fcrystalnet.core.snapshot;

/**
 * Everything the damage model and the placement rules need to know about one block, and
 * nothing else - no Minecraft type, so a whole region can be handwritten in a test.
 *
 * <p>Five kinds, because that is what the crystal pipeline actually distinguishes:
 * air (nothing there), replaceable (a placement can overwrite it), obsidian and bedrock
 * (valid crystal bases, and the only attenuators that matter for exposure), and other
 * (solid, breakable, irrelevant as a base).
 *
 * @param kind            which of the five cases this is
 * @param blastResistance the block's explosion resistance; 0 for air/replaceable
 * @param solid           whether it stops movement and placement
 */
public record BlockSnapshot(Kind kind, float blastResistance, boolean solid) {

    public enum Kind {
        AIR,
        REPLACEABLE,
        OBSIDIAN,
        BEDROCK,
        OTHER
    }

    public static final BlockSnapshot AIR = new BlockSnapshot(Kind.AIR, 0.0f, false);
    public static final BlockSnapshot REPLACEABLE = new BlockSnapshot(Kind.REPLACEABLE, 0.0f, false);
    /** Vanilla obsidian: resistance 1200, the standard crystal base. */
    public static final BlockSnapshot OBSIDIAN = new BlockSnapshot(Kind.OBSIDIAN, 1200.0f, true);
    /** Vanilla bedrock: resistance 3,600,000. A base we cannot place but can build on. */
    public static final BlockSnapshot BEDROCK = new BlockSnapshot(Kind.BEDROCK, 3_600_000.0f, true);
    /** Placeholder for solid breakable blocks tests do not want to name. */
    public static final BlockSnapshot STONE = new BlockSnapshot(Kind.OTHER, 6.0f, true);
    /** Catch-all: solid, weakly attenuating, never a base. */
    public static final BlockSnapshot OTHER = new BlockSnapshot(Kind.OTHER, 1.0f, true);

    /** A valid crystal base: the two blocks a crystal can sit on. */
    public boolean isBase() {
        return this.kind == Kind.OBSIDIAN || this.kind == Kind.BEDROCK;
    }

    /** True when a placement request may overwrite this block in place. */
    public boolean isReplaceable() {
        return this.kind == Kind.REPLACEABLE;
    }

    /** True when the block is "nothing" for placement purposes. */
    public boolean isEmpty() {
        return this.kind == Kind.AIR || this.kind == Kind.REPLACEABLE;
    }
}
