package com.github.luiox.fcrystalnet.core.snapshot;

import com.github.luiox.fcrystalnet.core.math.Vec3d;
import java.util.List;

/**
 * The shared target: its state now, and where it is going.
 *
 * <p>Perception owns target selection and prediction precisely so that every consumer -
 * the crystal decision, rendering, future modules - sees the <em>same</em> two points.
 * A module that recomputes prediction privately draws a box that disagrees with where
 * the crystal actually goes, and the user files a bug that cannot be found.
 *
 * @param current     the entity now
 * @param predicted   extrapolated future positions, one per step (0.5-tick steps in the
 *                    fabric implementation); same list reference as
 *                    {@code current.predicted()}
 * @param worthHitting cheap front-line judgment (alive, not in spawn, not a teammate)
 */
public record TargetSnapshot(EntitySnapshot current, List<Vec3d> predicted, boolean worthHitting) {

    public TargetSnapshot {
        predicted = predicted == null ? List.of() : List.copyOf(predicted);
    }

    /** The damage-model's preferred aim point: the first predicted position, or current. */
    public Vec3d aimPos() {
        return this.predicted.isEmpty() ? this.current.pos() : this.predicted.get(this.predicted.size() - 1);
    }
}
