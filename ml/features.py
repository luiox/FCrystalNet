"""Frame -> model tensors.

Terrain keeps its raw five-kind identity as one-hot channels. Entity channels are
continuous distance fields exp(-d/2) to the entity box surface (1.0 inside, smooth decay
outside) instead of 0/1 rasters: the scored position lives between cells, and a binary
box quantizes it away - the distance field keeps sub-cell information the argmax needs
for fine ranking. Channel 9 is the reach map (self-eye to cell-centre <= 4.5); it is
exact, recomputable in-game, and the network should spend its capacity on the damage
field, not on re-deriving a ruler. Channel 10 is the existing-crystal distance field -
the break side's subjects, whose damage is read off the same dmg head (detonating a
crystal at a cell is the cell's own explosion).

The crop never moves: it is target-centred by construction (origin = the target foot cell + (-6, -3, -6)), so "where is the target" is read off the aim channel directly.
"""

import numpy as np

N_KIND = 5
CH_AIM = 5       # distance field to the target hitbox at the scored (predicted) position
CH_CUR = 6       # distance field to the target hitbox now
CH_SELF = 7      # distance field to the self box
CH_OTHER = 8     # distance field to the nearest other entity box
CH_REACH = 9     # 0/1 placement-reach map (teacher-computed, exact at inference too)
CH_CRYSTAL = 10  # distance field to the nearest pre-placed end crystal (the break side)
CHANNELS = 11
AUX_DIM = 14
SIGMA = 2.0


def build_x(f):
    sx, sy, sz = f["dims"]
    shape = (sy, sz, sx)
    origin = f["origin"]
    x = np.zeros((CHANNELS, sy, sz, sx), np.float16)
    for k in range(N_KIND):
        x[k] = (f["kind"] == k)

    ents = f["ents"]
    tgt = ents[1]
    x[CH_SELF] = _dist_field(shape, ents[0]["box"], origin)
    x[CH_CUR] = _dist_field(shape, tgt["box"], origin)
    if tgt["preds"]:
        p = np.array(tgt["preds"][-1])
        shift = p - tgt["pos"]
        aim_box = tgt["box"].copy()
        aim_box[:3] += shift
        aim_box[3:] += shift
        x[CH_AIM] = _dist_field(shape, aim_box, origin)
    others = [e for e in ents[2:] if e["kind"] != 3]
    if others:
        best = _dist_field(shape, others[0]["box"], origin).astype(np.float32)
        for e in others[1:]:
            np.maximum(best, _dist_field(shape, e["box"], origin), out=best)
        x[CH_OTHER] = best.astype(np.float16)
    crystals = [e for e in ents[2:] if e["kind"] == 3]
    if crystals:
        best = _dist_field(shape, crystals[0]["box"], origin).astype(np.float32)
        for e in crystals[1:]:
            np.maximum(best, _dist_field(shape, e["box"], origin), out=best)
        x[CH_CRYSTAL] = best.astype(np.float16)
    x[CH_REACH] = f["reach"]
    return x


def _dist_field(shape, box, origin, sigma=SIGMA):
    """exp(-d / sigma) per voxel centre, d = distance to the box surface (0 inside)."""
    sy, sz, sx = shape
    ox, oy, oz = origin
    cx = (np.arange(sx, dtype=np.float32) + 0.5 + ox)[None, None, :]
    cy = (np.arange(sy, dtype=np.float32) + 0.5 + oy)[:, None, None]
    cz = (np.arange(sz, dtype=np.float32) + 0.5 + oz)[None, :, None]
    dx = np.maximum(np.maximum(box[0] - cx, cx - box[3]), 0.0)
    dy = np.maximum(np.maximum(box[1] - cy, cy - box[4]), 0.0)
    dz = np.maximum(np.maximum(box[2] - cz, cz - box[5]), 0.0)
    d = np.sqrt(dx * dx + dy * dy + dz * dz)
    return np.exp(-d / sigma).astype(np.float16)


def build_aux(f):
    return np.array([
        f["tvel"][0] / 0.35, f["tvel"][1] / 0.35, f["tvel"][2] / 0.35,
        f["svel"][0] / 0.35, f["svel"][1] / 0.35, f["svel"][2] / 0.35,
        f["aim_steps"] / 5.0,
        f["t_armor"] / 20.0, f["t_tough"] / 20.0, f["t_prot"] / 20.0, f["t_health"] / 20.0,
        f["s_armor"] / 20.0, f["s_prot"] / 20.0,
        f["difficulty"] / 3.0,
    ], np.float32)


def build_dataset(frames):
    """Stacked tensors; X stays float16 (distance fields), batch conversion to float32
    happens in the training loop."""
    x = np.stack([build_x(f) for f in frames])
    aux = np.stack([build_aux(f) for f in frames])
    place = np.stack([np.maximum(f["valid"], f["needs_base"]) for f in frames]).astype(np.uint8)
    brk = np.stack([f["breakable"] for f in frames]).astype(np.uint8)
    dmg = np.stack([f["dmg"] for f in frames]).astype(np.float32)
    self_dmg = np.stack([f["self_dmg"] for f in frames]).astype(np.float32)
    reach = np.stack([f["reach"] for f in frames]).astype(np.uint8)
    seeds = np.array([f["seed"] for f in frames], np.int64)
    presets = np.array([f["preset"] for f in frames], np.int64)
    s_health = np.array([f["s_health"] for f in frames], np.float32)
    return {
        "x": x, "aux": aux, "place": place, "break": brk, "dmg": dmg,
        "self_dmg": self_dmg, "reach": reach, "seeds": seeds, "presets": presets,
        "s_health": s_health,
    }
