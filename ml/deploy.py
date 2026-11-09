"""Export a trained run to ONNX and get the deployment numbers through ORT.

The numbers the pocmod cares about, straight through onnxruntime CPU (not torch):
single-thread latency (the in-game budget lives on one core beside Minecraft) and
the full dual-metric accuracy - placement top5/regret, break top1/regret - so the
export step is never trusted on a loss curve alone.

int8 experiments live in quantize.py and stay there: v1 concluded negative for
Conv3d (ORT has no 3-D int8 conv kernels), so this script only ships fp32.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from features import build_dataset
from model import CrystalNet
from read_frames import load_dir


def export_onnx(run: Path) -> Path:
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = CrystalNet(channels=cfg["channels"], aux_dim=cfg["aux_dim"],
                   width=cfg["width"], factorized=cfg.get("factorized", False))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    x = torch.zeros(1, cfg["channels"], 7, 13, 13)
    a = torch.zeros(1, cfg["aux_dim"])
    out = run / "model.onnx"
    torch.onnx.export(m, (x, a), out, input_names=["x", "aux"],
                      output_names=["dmg", "valid"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "batch"}, "aux": {0: "batch"},
                                    "dmg": {0: "batch"}, "valid": {0: "batch"}})
    print(f"exported: {out} ({out.stat().st_size / 1e6:.2f} MB, "
          f"params epoch {ckpt.get('epoch', '?')})")
    return out


def bench(path: Path, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    n_ch = sess.get_inputs()[0].shape[1]
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
    """Dual metrics through the ORT session: placement (oracle + model gates) and break."""
    x, aux = val["x"], val["aux"]
    place, brk = val["place"], val["break"]
    dmg, self_dmg = val["dmg"], val["self_dmg"]
    reach = val["reach"]
    s_health = val["s_health"][:, None, None, None]
    veto = self_dmg < s_health - 1.0
    oracle = (place > 0) & (reach > 0) & veto

    out = {}
    dmg_pred = np.empty(len(x), object)
    valid_logit = np.empty(len(x), object)
    for i in range(0, len(x), 512):
        d, v = sess.run(None, {"x": x[i:i + 512].astype(np.float32),
                               "aux": aux[i:i + 512].astype(np.float32)})
        for b in range(d.shape[0]):
            dmg_pred[i + b] = d[b]
            valid_logit[i + b] = v[b]

    def place_score(gate, tau=None):
        tops1, tops5, regs = [], [], []
        for b in range(len(x)):
            g = gate[b] if tau is None else ((valid_logit[b] > tau) & (reach[b] > 0) & veto[b])
            if not g.any():
                continue
            dd = np.where(g, dmg_pred[b], -1e9)
            lbl = np.where(g, dmg[b], -1e9)
            best = int(lbl.flatten().argmax())
            top5 = np.argpartition(dd.flatten(), -5)[-5:]
            tops1.append(1.0 if best == int(dd.flatten().argmax()) else 0.0)
            tops5.append(1.0 if best in top5 else 0.0)
            regs.append(float(dmg[b].flatten()[best] - dmg[b].flatten()[dd.flatten().argmax()]))
        reg = np.array(regs)
        return {"n": len(tops1), "top1": float(np.mean(tops1)), "top5": float(np.mean(tops5)),
                "regret_mean": float(reg.mean()), "regret_p90": float(np.percentile(reg, 90))}

    def break_score():
        hits, regs = [], []
        gate = (brk > 0) & (reach > 0) & veto
        for b in range(len(x)):
            g = gate[b]
            if not g.any():
                continue
            dd = np.where(g, dmg_pred[b], -1e9)
            lbl = np.where(g, dmg[b], -1e9)
            best = int(lbl.flatten().argmax())
            pick = int(dd.flatten().argmax())
            hits.append(1.0 if best == pick else 0.0)
            regs.append(float(dmg[b].flatten()[best] - dmg[b].flatten()[pick]))
        reg = np.array(regs)
        return {"n": len(hits), "top1": float(np.mean(hits)),
                "regret_mean": float(reg.mean()) if len(reg) else float("nan")}

    out["oracle_gated"] = place_score(oracle)
    out["model_gated"] = place_score(None, tau=0.5)
    out["break_gated"] = break_score()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/w12v2long")
    ap.add_argument("--data", default="data/ds")
    args = ap.parse_args()

    run = Path(args.run)
    onnx_path = run / "model.onnx"
    if not onnx_path.exists():
        export_onnx(run)

    frames, _meta = load_dir(args.data)
    ds = build_dataset(frames)
    is_val = (np.uint64(ds["seeds"]) * np.uint64(2654435761) % 100) < 12
    val = {k: v[is_val] for k, v in ds.items()}
    print(f"val frames: {len(val['x'])}")

    for threads in (1, 4):
        sess, p50, p99 = bench(onnx_path, threads)
        print(f"ORT {threads} thread(s): p50 {p50:.2f} ms  p99 {p99:.2f} ms")

    sess, _, _ = bench(onnx_path, 1)
    acc = accuracy(sess, val)
    print("\n| task | n | top1 | top5 | regret mean | regret p90 |")
    print("|---|---|---|---|---|---|")
    for name in ("oracle_gated", "model_gated", "break_gated"):
        r = acc[name]
        top5 = f"{r['top5']:.3f}" if "top5" in r else "-"
        p90 = f"{r['regret_p90']:.3f}" if "regret_p90" in r else "-"
        print(f"| {name} | {r['n']} | {r['top1']:.3f} | {top5} | {r['regret_mean']:.3f} | {p90} |")


if __name__ == "__main__":
    main()
