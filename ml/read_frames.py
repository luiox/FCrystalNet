"""Reader for .ncds frames.

Layout reference: docs/dataset-schema.md. Every frame is self-describing (magic +
schemaVer + payloadLen), so torn shards resync at the next record instead of poisoning
the rest of the file, and mismatched schema versions are rejected.
"""

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NCDS"
# BlockSnapshot.Kind ordinal order - do not reorder, the terrain bytes ARE these numbers.
KIND_NAMES = ["AIR", "REPLACEABLE", "OBSIDIAN", "BEDROCK", "OTHER"]
# Entity kind codes.
ENT_SELF = 0
ENT_TARGET = 1
ENT_OTHER = 2
ENT_CRYSTAL = 3
PRESET_NAMES = ["open_field", "box_fight", "hole", "nether_roof", "pillar_city", "blobs", "mixed"]


def iter_frames(path):
    data = Path(path).read_bytes()
    n = len(data)
    off = 0
    while off < n:
        if data[off:off + 4] != MAGIC:
            raise ValueError(f"bad magic at {off} in {path}")
        ver, plen = struct.unpack_from("<II", data, off + 4)
        payload = data[off + 12:off + 12 + plen]
        if len(payload) != plen:
            raise ValueError(f"truncated frame at {off} in {path}")
        yield decode(payload, ver)
        off += 12 + plen


def decode(p, ver=2):
    o = 0
    (seed,) = struct.unpack_from("<q", p, o); o += 8
    preset, diff, aim_steps, ent_count = struct.unpack_from("<4B", p, o); o += 4
    tvel = np.array(struct.unpack_from("<3f", p, o), np.float32); o += 12
    svel = np.array(struct.unpack_from("<3f", p, o), np.float32); o += 12
    t_armor, t_tough, t_prot, t_health = struct.unpack_from("<4f", p, o); o += 16
    s_armor, s_tough, s_prot, s_health = struct.unpack_from("<4f", p, o); o += 16
    ox, oy, oz = struct.unpack_from("<3i", p, o); o += 12
    sx, sy, sz = struct.unpack_from("<3B", p, o); o += 3
    cells = sx * sy * sz

    ents = []
    for _ in range(ent_count):
        (kind,) = struct.unpack_from("<B", p, o); o += 1
        pos = struct.unpack_from("<3d", p, o); o += 24
        box = struct.unpack_from("<6d", p, o); o += 48
        stats = struct.unpack_from("<6f", p, o); o += 24
        (npred,) = struct.unpack_from("<B", p, o); o += 1
        preds = []
        for _i in range(npred):
            preds.append(struct.unpack_from("<3d", p, o)); o += 24
        ents.append({"kind": kind, "pos": np.array(pos), "box": np.array(box),
                     "stats": stats, "preds": preds})

    def u8():
        nonlocal o
        a = np.frombuffer(p, np.uint8, cells, o).reshape(sy, sz, sx).copy()
        o += cells
        return a

    kind = u8(); valid = u8(); needs_base = u8(); reach = u8()
    if ver >= 2:
        breakable = u8()
    else:
        breakable = np.zeros((sy, sz, sx), np.uint8)

    def f32():
        nonlocal o
        a = np.frombuffer(p, "<f4", cells, o).reshape(sy, sz, sx).copy()
        o += 4 * cells
        return a

    dmg = f32(); self_dmg = f32()
    assert o == len(p), f"frame payload over/under-run: {o} != {len(p)}"

    return {
        "schema": ver,
        "seed": seed, "preset": preset, "difficulty": diff, "aim_steps": aim_steps,
        "tvel": tvel, "svel": svel,
        "t_armor": t_armor, "t_tough": t_tough, "t_prot": t_prot, "t_health": t_health,
        "s_armor": s_armor, "s_tough": s_tough, "s_prot": s_prot, "s_health": s_health,
        "origin": (ox, oy, oz), "dims": (sx, sy, sz), "ents": ents,
        "kind": kind, "valid": valid, "needs_base": needs_base, "reach": reach,
        "breakable": breakable, "dmg": dmg, "self_dmg": self_dmg,
    }


def load_dir(d):
    files = sorted(Path(d).glob("shard_*.ncds"))
    if not files:
        raise FileNotFoundError(f"no shards under {d}")
    frames = []
    for f in files:
        frames.extend(iter_frames(f))
    meta = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
    return frames, meta
