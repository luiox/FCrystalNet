"""Quantization experiments: does int8 actually buy anything for a tiny Conv3d on CPU?

Three candidates against the fp32 baseline:
  - dynamic int8 (weights int8, activations quantized at runtime)
  - static QDQ int8 (calibrated on real frames, per-channel weights)
  - both measured for latency AND accuracy - a quantized model that loses top5 is not
    a win at any speed.

Caveat this script exists to test: ONNX QLinearConv/ConvInteger are 2-D ops, so 3-D
convs may refuse to quantize and silently stay float. Never trust the file name - read
the latency.
"""

import time
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from read_frames import load_dir
from features import build_dataset

FP32 = Path("runs/v1/model.onnx")
DYN = Path("runs/v1/model.int8dyn.onnx")
STATIC = Path("runs/v1/model.int8static.onnx")


def export_fp32():
    import onnx
    from model import CrystalNet
    ckpt = torch.load("runs/v1/model.pt", weights_only=False)
    m = CrystalNet(channels=ckpt["config"]["channels"], aux_dim=ckpt["config"]["aux_dim"],
                   width=ckpt["config"]["width"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    x = torch.zeros(1, ckpt["config"]["channels"], 7, 13, 13)
    a = torch.zeros(1, 14)
    torch.onnx.export(m, (x, a), FP32, input_names=["x", "aux"],
                      output_names=["dmg", "valid"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "batch"}, "aux": {0: "batch"},
                                    "dmg": {0: "batch"}, "valid": {0: "batch"}})
    print(f"fp32 onnx: {FP32.stat().st_size / 1e6:.2f} MB")


class Reader(CalibrationDataReader):

    def __init__(self, xs, aus):
        self.items = [{"x": x[None], "aux": a[None]} for x, a in zip(xs, aus)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        it = self.items[self.i]
        self.i += 1
        return it


def make_quantized(frames):
    ds = build_dataset(frames[:400])
    calib = [(ds["x"][i].astype(np.float32), ds["aux"][i]) for i in range(len(ds["x"]))]

    quantize_dynamic(FP32, DYN, weight_type=QuantType.QInt8)
    quantize_static(FP32, STATIC, Reader([c[0] for c in calib], [c[1] for c in calib]),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    print(f"dynamic int8: {DYN.stat().st_size / 1e6:.2f} MB")
    print(f"static QDQ:   {STATIC.stat().st_size / 1e6:.2f} MB")


def bench(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    c = sess.get_inputs()[0].shape
    n_ch = c[1] if isinstance(c[1], int) else 10
    xn = np.zeros((1, n_ch, 7, 13, 13), np.float32)
    an = np.zeros((1, 14), np.float32)
    for _ in range(30):
        sess.run(None, {"x": xn, "aux": an})
    ts = []
    for _ in range(500):
        t0 = time.perf_counter()
        sess.run(None, {"x": xn, "aux": an})
        ts.append((time.perf_counter() - t0) * 1e3)
    return sess, float(np.percentile(ts, 50)), float(np.percentile(ts, 99))


def accuracy(sess, val):
    """Oracle-gated top5 / regret on the val split, straight through the ORT session."""
    x = val["x"]
    aux = val["aux"]
    place = val["place"]
    dmg = val["dmg"]
    self_dmg = val["self_dmg"]
    gate_all = (place > 0) & (self_dmg < val["s_health"][:, None, None, None] - 1.0)
    tops5, regrets = [], []
    for i in range(0, len(x), 512):
        d, _v = sess.run(None, {"x": x[i:i + 512].astype(np.float32),
                                "aux": aux[i:i + 512]})
        for b in range(d.shape[0]):
            g = gate_all[i + b]
            if not g.any():
                continue
            dd = np.where(g, d[b], -1e9)
            lbl = np.where(g, dmg[i + b], -1e9)
            best = int(lbl.flatten().argmax())
            top5 = np.argpartition(dd.flatten(), -5)[-5:]
            tops5.append(1.0 if best in top5 else 0.0)
            pick = dd.flatten().argmax()
            regrets.append(float(dmg[i + b].flatten()[best] - dmg[i + b].flatten()[pick]))
    return float(np.mean(tops5)), float(np.mean(regrets))


def main():
    frames, _meta = load_dir("data/ds")
    if not FP32.exists():
        export_fp32()
    if not DYN.exists() or not STATIC.exists():
        make_quantized(frames)

    _, meta = load_dir("data/ds")
    ds = build_dataset(frames)
    is_val = (np.uint64(ds["seeds"]) * np.uint64(2654435761) % 100) < 12
    val = {k: v[is_val] for k, v in ds.items()}

    rows = []
    for name, path in (("fp32", FP32), ("int8 dynamic", DYN), ("int8 static QDQ", STATIC)):
        sess, p50, p99 = bench(path)
        top5, reg = accuracy(sess, val)
        rows.append((name, path.stat().st_size / 1e6, p50, p99, top5, reg))

    print("\n| model | size MB | p50 ms | p99 ms | val top5 | regret mean |")
    print("|---|---|---|---|---|---|")
    for name, mb, p50, p99, top5, reg in rows:
        print(f"| {name} | {mb:.2f} | {p50:.2f} | {p99:.2f} | {top5:.3f} | {reg:.3f} |")


if __name__ == "__main__":
    main()
