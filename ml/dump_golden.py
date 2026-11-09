"""Golden fixtures for the pocmod's offline tests.

Picks a deterministic spread of frames from the dataset directory (crystal-rich, evenly spaced, plus a
couple of bare ones), rebuilds the model tensors exactly as training saw them (the fp16
rounding included), runs the exported ONNX through ORT CPU single-thread, and writes one
little-endian blob the Java tests decode and compare against:

    magic "FCG1" | u32 frameCount
    per frame:
        u32 payloadLen | bytes payload (.ncds frame payload, verbatim)
        f32 x[11*1183] | f32 aux[14] | f32 dmg[1183] | f32 valid[1183]

The payload carries everything FeatureBuilder needs (kinds, entities, preds, stats), so
the Java side rebuilds the features from it and the two implementations must agree to
1e-3; the ORT outputs pin the DLL's inference to the same bar. Together the two tests
chain: python features == java features, python ORT == crystalnet.dll, therefore the DLL
on java features is the model.
"""

import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort

from features import build_aux, build_x
from read_frames import decode

DATA = Path(__file__).parent / "data" / "ds"
RUN = Path(__file__).parent / "runs" / "w24v2gpu"
OUT = Path(__file__).parent.parent / "pocmod" / "src" / "test" / "resources" / "golden.bin"
N_CRYSTAL = 6
N_BARE = 2
MAGIC = b"FCG1"


def iter_records(path):
    """(payload, decoded) per record - unlike iter_frames, the payload survives."""
    data = path.read_bytes()
    off = 0
    while off < len(data):
        if data[off:off + 4] != b"NCDS":
            raise ValueError(f"bad magic at {off} in {path}")
        ver, plen = struct.unpack_from("<II", data, off + 4)
        payload = data[off + 12:off + 12 + plen]
        if len(payload) != plen:
            raise ValueError(f"truncated frame at {off} in {path}")
        yield payload, decode(payload, ver)
        off += 12 + plen


def spread(indices, n):
    """n indices evenly spaced across the list (deterministic, first to last)."""
    if len(indices) <= n:
        return list(indices)
    return [indices[round(i * (len(indices) - 1) / (n - 1))] for i in range(n)]


def main():
    records = []
    for path in sorted(DATA.glob("shard_*.ncds")):
        records.extend(iter_records(path))
    crystal_idx = [i for i, (_, f) in enumerate(records)
                   if any(e["kind"] == 3 for e in f["ents"])]
    bare_idx = [i for i, (_, f) in enumerate(records)
                if not any(e["kind"] == 3 for e in f["ents"])]
    picked = spread(crystal_idx, N_CRYSTAL) + spread(bare_idx, N_BARE)
    print(f"loaded {len(records)} frames; picked {len(picked)} "
          f"({len(spread(crystal_idx, N_CRYSTAL))} crystal + "
          f"{len(spread(bare_idx, N_BARE))} bare), seeds: "
          f"{[records[i][1]['seed'] for i in picked]}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(RUN / "model.onnx"), so,
                                providers=["CPUExecutionProvider"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as w:
        w.write(MAGIC)
        w.write(struct.pack("<I", len(picked)))
        for i in picked:
            payload, f = records[i]
            # The exact training-time tensors: fp16 distance fields, fp32 aux.
            x = build_x(f).astype(np.float32)
            aux = build_aux(f).astype(np.float32)
            dmg, valid = sess.run(None, {"x": x[None], "aux": aux[None]})
            w.write(struct.pack("<I", len(payload)))
            w.write(payload)
            w.write(x.tobytes())
            w.write(aux.tobytes())
            w.write(np.asarray(dmg, np.float32).tobytes())
            w.write(np.asarray(valid, np.float32).tobytes())
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, "
          f"{sum(1 for _ in picked)} frames, model {RUN / 'model.onnx'})")


if __name__ == "__main__":
    main()
