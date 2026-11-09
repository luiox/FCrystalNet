"""Evaluate a trained run against the teacher on the held-out split.

Reports what actually decides whether the model is deployable: top-k recall (the exact
re-rank rescues everything in the top-5), damage regret per quantile, pick validity, and
single-thread latency - the number the 50ms budget cares about.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from features import build_dataset
from model import CrystalNet
from read_frames import PRESET_NAMES, load_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ds")
    ap.add_argument("--run", default="runs/v1")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="validity logit threshold for model-side gating")
    args = ap.parse_args()

    ckpt = torch.load(Path(args.run) / "model.pt", map_location="cpu", weights_only=False)
    model = CrystalNet(channels=ckpt["config"].get("channels", 10),
                       aux_dim=ckpt["config"]["aux_dim"], width=ckpt["config"]["width"],
                       factorized=ckpt["config"].get("factorized", False))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    frames, meta = load_dir(args.data)
    ds = build_dataset(frames)
    is_val = (np.uint64(ds["seeds"]) * np.uint64(2654435761) % 100) < 12
    val = {k: v[is_val] for k, v in ds.items()}
    x = torch.from_numpy(val["x"])
    aux = torch.from_numpy(val["aux"])

    with torch.no_grad():
        dmg_pred, valid_logit = [], []
        for i in range(0, len(x), 512):
            d, v = model(x[i:i + 512].float(), aux[i:i + 512])
            dmg_pred.append(d)
            valid_logit.append(v)
        dmg_pred = torch.cat(dmg_pred)
        valid_logit = torch.cat(valid_logit)

    place = torch.from_numpy(val["place"])
    reach = torch.from_numpy(val["reach"])
    dmg_lbl = torch.from_numpy(val["dmg"])
    self_dmg = torch.from_numpy(val["self_dmg"])
    s_health = torch.from_numpy(val["s_health"])[:, None, None, None]
    presets = torch.from_numpy(val["presets"])

    veto = self_dmg < s_health - 1.0
    gates = {
        "oracle_gated": (place > 0) & (reach > 0) & veto,
        "model_gated": (valid_logit > args.tau) & (reach > 0) & veto,
    }
    results = {}
    for name, gate in gates.items():
        results[name] = score(gate, dmg_pred, dmg_lbl, place, presets)

    # The break side: which existing crystal to detonate. Same dmg head, ranked over the
    # breakable cells only; gate = crystal exists (perception knows this exactly) + the
    # self-damage hard veto first. The
    # reach map is the placement gate - attack reach is shorter, so this is the
    # optimistic bound.
    brk = torch.from_numpy(val["break"])
    break_gate = (brk > 0) & (reach > 0) & veto
    results["break_gated"] = score_break(break_gate, dmg_pred, dmg_lbl)

    lat = latency(model, x.shape[1:])
    rows = [(n, r) for n, r in results.items() if n != "break_gated"]
    break_row = results["break_gated"]
    print_report(rows, lat, len(val["x"]), results, break_row)
    write_report(Path(args.run) / "report.md", rows, lat, len(val["x"]), meta, results,
                 break_row)


def score(gate, dmg_pred, dmg_lbl, place, presets):
    tops1, tops5, regrets = [], [], []
    per_preset = {p: [] for p in range(len(PRESET_NAMES))}
    n_picked_valid = 0
    n_frames = 0
    for b in range(dmg_pred.shape[0]):
        g = gate[b]
        if not g.any():
            continue
        n_frames += 1
        p = torch.where(g, dmg_pred[b], torch.full_like(dmg_pred[b], -1e9))
        lbl = torch.where(g, dmg_lbl[b], torch.full_like(dmg_lbl[b], -1e9))
        best_idx = int(lbl.flatten().argmax())
        top5 = torch.topk(p.flatten(), 5).indices.tolist()
        hit1 = 1.0 if top5[0] == best_idx else 0.0
        tops1.append(hit1)
        tops5.append(1.0 if best_idx in top5 else 0.0)
        regrets.append(float(dmg_lbl[b].flatten()[best_idx]
                             - dmg_lbl[b].flatten()[top5[0]]))
        per_preset[int(presets[b])].append(hit1)
        n_picked_valid += 1 if place[b].flatten()[top5[0]] > 0 else 0
    reg = np.array(regrets)
    return {
        "n": n_frames,
        "top1": float(np.mean(tops1)),
        "top5": float(np.mean(tops5)),
        "regret_mean": float(reg.mean()),
        "regret_p50": float(np.percentile(reg, 50)),
        "regret_p90": float(np.percentile(reg, 90)),
        "regret_max": float(reg.max()),
        "valid_rate": n_picked_valid / max(n_frames, 1),
        "top1_per_preset": {PRESET_NAMES[p]: float(np.mean(v)) if v else None
                            for p, v in per_preset.items()},
    }


def score_break(gate, dmg_pred, dmg_lbl):
    """Break metrics over the gated breakable cells; candidates are few (0-3), so only
    argmax agreement and regret matter - there is no top-5 to rescue a miss."""
    hits, regs = [], []
    for b in range(dmg_pred.shape[0]):
        g = gate[b]
        if not g.any():
            continue
        lbl = torch.where(g, dmg_lbl[b], torch.full_like(dmg_pred[b], -1e9))
        pred = torch.where(g, dmg_pred[b], torch.full_like(dmg_pred[b], -1e9))
        best = int(lbl.flatten().argmax())
        pick = int(pred.flatten().argmax())
        hits.append(1.0 if best == pick else 0.0)
        regs.append(float(dmg_lbl[b].flatten()[best] - dmg_lbl[b].flatten()[pick]))
    reg = np.array(regs)
    return {
        "n": len(hits),
        "top1": float(np.mean(hits)) if hits else float("nan"),
        "regret_mean": float(reg.mean()) if len(reg) else float("nan"),
        "regret_p90": float(np.percentile(reg, 90)) if len(reg) else float("nan"),
        "regret_max": float(reg.max()) if len(reg) else float("nan"),
    }


def latency(model, in_shape):
    torch.set_num_threads(1)
    x = torch.zeros(1, *in_shape)
    a = torch.zeros(1, 14)
    with torch.no_grad():
        for _ in range(20):
            model(x, a)
        ts = []
        for _ in range(300):
            t0 = time.perf_counter()
            model(x, a)
            ts.append((time.perf_counter() - t0) * 1e3)
    torch.set_num_threads(torch.get_num_threads())
    return {"p50_ms": float(np.percentile(ts, 50)),
            "p99_ms": float(np.percentile(ts, 99))}


def fmt_table(rows):
    keys = ["n", "top1", "top5", "regret_mean", "regret_p50", "regret_p90",
            "regret_max", "valid_rate"]
    head = "| gating | " + " | ".join(keys) + " |"
    sep = "|---" * (len(keys) + 1) + "|"
    lines = [head, sep]
    for name, r in rows:
        lines.append("| " + name + " | " + " | ".join(
            f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
    return "\n".join(lines)


def fmt_per_preset(r):
    lines = ["| preset | top1 |", "|---|---|"]
    for name, v in r["top1_per_preset"].items():
        lines.append(f"| {name} | {'n/a' if v is None else f'{v:.3f}'} |")
    return "\n".join(lines)


def print_report(rows, lat, n_val, results, break_row):
    print(f"\nval frames: {n_val}")
    print(fmt_table(rows))
    print(f"latency (1 thread): p50 {lat['p50_ms']:.2f}ms  p99 {lat['p99_ms']:.2f}ms")
    print(f"break (gated breakable cells, n={break_row['n']}): "
          f"top1 {break_row['top1']:.3f}  regret mean {break_row['regret_mean']:.3f} "
          f"p90 {break_row['regret_p90']:.3f} max {break_row['regret_max']:.3f}")
    print("\nper-preset top1 (oracle-gated):")
    print(fmt_per_preset(results["oracle_gated"]))


def write_report(path, rows, lat, n_val, meta, results, break_row):
    lines = [
        "# Crystal placement model - POC eval",
        "",
        f"- data: {meta['frames']} frames, val split {n_val}",
        f"- latency (CPU, 1 thread): **p50 {lat['p50_ms']:.2f} ms**, p99 {lat['p99_ms']:.2f} ms",
        "",
        fmt_table(rows),
        "",
        f"Break (gated breakable cells, n={break_row['n']}): "
        f"top1 {break_row['top1']:.3f}, regret mean {break_row['regret_mean']:.3f}, "
        f"p90 {break_row['regret_p90']:.3f}, max {break_row['regret_max']:.3f}",
        "",
        "Per-preset top1 (oracle-gated):",
        "",
        fmt_per_preset(results["oracle_gated"]),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
