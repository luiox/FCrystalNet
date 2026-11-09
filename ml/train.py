"""Train CrystalNet on teacher-labelled frames.

The rotation/mirror augmentation exploits the problem's exact symmetry (the world has no
preferred compass direction), so it multiplies the data 8x for free - but the aux vector
must rotate with the crop, or the model learns that velocity points nowhere. The asserts
below pin that correspondence at startup.

Device: CUDA when available (RTX 40-series trains this net in minutes), CPU otherwise.
Checkpointing: every epoch writes `ckpt.pt` - model + optimizer + scheduler + RNG state
+ history - and `--resume <runs-dir>` continues an interrupted run from it exactly, so a
long run can hop machines mid-schedule. `model.pt` stays the best-val export (CPU-side
tensors, loads anywhere) for eval.py/deploy.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from features import CHANNELS, build_dataset
from model import CrystalNet, count_params
from read_frames import load_dir


def rot_k(t, k):
    """Rotate the (Z, X) plane by k*90 degrees, matching (x, z) -> (-z, x)."""
    for _ in range(k % 4):
        t = t.transpose(-1, -2).flip(-1)
    return t


def rot_vel(vx, vz, k):
    for _ in range(k % 4):
        vx, vz = -vz, vx
    return vx, vz


def augment_aux(aux, k, flip):
    """Apply the same spatial transform as the crop to every velocity column.

    Layout: aux[0:3] = target vel, aux[3:6] = self vel. Miss one and the augmentation
    teaches the net that terrain rotates while inertia does not - a physics-impossible
    pseudo-correlation, and it poisons the aux branch even if the column is zero today.
    """
    aux = aux.clone()
    if flip:
        aux[:, 0] = -aux[:, 0]
        aux[:, 3] = -aux[:, 3]
    for _ in range(k % 4):
        vx, vz = aux[:, 0].clone(), aux[:, 2].clone()
        aux[:, 0], aux[:, 2] = -vz, vx
        vx, vz = aux[:, 3].clone(), aux[:, 5].clone()
        aux[:, 3], aux[:, 5] = -vz, vx
    return aux


def _selfcheck_augment():
    t = torch.zeros(1, 1, 7, 13, 13)
    t[0, 0, 3, 1, 2] = 1.0  # (y=3, z=1, x=2)
    r = rot_k(t, 1)
    assert r[0, 0, 3, 2, 11] == 1.0, "rotation must map (x=2,z=1) -> (x=11,z=2), y fixed"
    assert r.sum() == 1.0
    aux = torch.zeros(2, 14)
    aux[0, 0], aux[0, 2] = 0.3, 0.4   # target vel x/z
    aux[1, 3], aux[1, 5] = 0.1, 0.2   # self vel x/z
    a = augment_aux(aux, 1, False)
    assert abs(a[0, 0] + 0.4) < 1e-6 and abs(a[0, 2] - 0.3) < 1e-6, "tvel must rotate"
    assert abs(a[1, 3] + 0.2) < 1e-6 and abs(a[1, 5] - 0.1) < 1e-6, "svel must rotate too"
    a = augment_aux(aux, 0, True)
    assert abs(a[0, 0] + 0.3) < 1e-6 and abs(a[1, 3] + 0.1) < 1e-6, "flip negates x for both"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ds")
    ap.add_argument("--runs", default="runs/v2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--factorized", action="store_true",
                    help="R(2+1)D-style (1,3,3)+(3,1,1) convs")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="",
                    help="cuda / cpu; default auto (cuda when available)")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable bf16 autocast on CUDA")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from <runs>/ckpt.pt")
    args = ap.parse_args()

    _selfcheck_augment()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    out = Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed shapes all the way down
    amp = dev.type == "cuda" and not args.no_amp
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "")
          + f" | amp(bf16): {amp}", flush=True)

    frames, meta = load_dir(args.data)
    print(f"frames: {len(frames)} (meta: {meta['frames']})", flush=True)
    print("building tensors (distance fields, one pass)...", flush=True)
    ds = build_dataset(frames)

    is_val = (np.uint64(ds["seeds"]) * np.uint64(2654435761) % 100) < 12
    val = {k: v[is_val] for k, v in ds.items()}
    trn = {k: v[~is_val] for k, v in ds.items()}
    print(f"train {len(trn['x'])} / val {len(val['x'])}", flush=True)

    model = CrystalNet(channels=CHANNELS, aux_dim=ds["aux"].shape[1], width=args.width,
                       factorized=args.factorized).to(dev)
    print(f"params: {count_params(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Per-frame damage normalisation: the network learns the field's shape; the absolute
    # half-heart scale rides in via aux (health/armor) only as far as it shifts argmax.
    # The supervised cell set is place ∪ breakable - detonating a pre-placed crystal at
    # a cell is that cell's own explosion, so one dmg head serves both tasks.
    def norm_dmg(dmg, sup):
        dmax = np.where(sup > 0, dmg, -np.inf).reshape(len(dmg), -1).max(1)
        scale = np.maximum(dmax, 0)
        # A worthless board (best supervised cell < 0.5 half-hearts) gets a zero target
        # and drops out of the damage loss - dividing by ~1e-3 would fabricate values.
        has = dmax > 0.5
        y = np.where(has[:, None, None, None], dmg / (scale[:, None, None, None] + 1e-3), 0.0)
        return y.astype(np.float32), has

    trn["sup"] = np.maximum(trn["place"], trn["break"])
    val["sup"] = np.maximum(val["place"], val["break"])
    trn["ynorm"], trn["has_dmg"] = norm_dmg(trn["dmg"], trn["sup"])
    val["ynorm"], _ = norm_dmg(val["dmg"], val["sup"])

    def to_t(d):
        return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in d.items()}

    trn_t = to_t(trn)
    val_t = to_t(val)

    n = len(trn_t["x"])
    hist = []
    best_top1 = -1.0
    start_epoch = 0

    # Resume: the full training state lives in ckpt.pt (model + optim + sched + RNG +
    # history), rewritten atomically each epoch - a run can hop machines mid-schedule.
    ckpt_path = out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_top1 = ck["best_top1"]
        hist = ck["hist"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        np.random.set_state(ck["np_rng"])
        if ck.get("cuda_rng") is not None and dev.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        print(f"resumed: epoch {ck['epoch']} -> starting {start_epoch}, "
              f"best_top1 {best_top1:.3f}", flush=True)

    def save_ckpt(epoch):
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({
            "epoch": epoch,
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "best_top1": best_top1,
            "hist": hist,
            "torch_rng": torch.get_rng_state().cpu(),
            "np_rng": np.random.get_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if dev.type == "cuda" else None),
            "config": {"channels": CHANNELS, "aux_dim": ds["aux"].shape[1],
                       "width": args.width, "factorized": args.factorized,
                       "epochs": args.epochs, "batch": args.batch, "lr": args.lr},
        }, tmp)
        tmp.replace(ckpt_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=dev)
        tot = {"loss": 0.0, "valid": 0.0, "dmg": 0.0}
        nb = 0
        n_batches = (n + args.batch - 1) // args.batch
        for bi, i in enumerate(range(0, n, args.batch)):
            idx = perm[i:i + args.batch]
            k = int(torch.randint(0, 4, (1,)))
            flip = bool(torch.randint(0, 2, (1,)))
            x = trn_t["x"][idx].float()
            aux = augment_aux(trn_t["aux"][idx], k, flip)
            y_dmg = trn_t["ynorm"][idx].float()
            place = trn_t["place"][idx].float()
            brk = trn_t["break"][idx].float()
            if flip:
                x = x.flip(-1)
                y_dmg = y_dmg.flip(-1)
                place = place.flip(-1)
                brk = brk.flip(-1)
            if k:
                x = rot_k(x, k)
                y_dmg = rot_k(y_dmg, k)
                place = rot_k(place, k)
                brk = rot_k(brk, k)

            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                dmg_pred, valid_logit = model(x, aux)
                pos = place.sum().clamp(min=1)
                pw = ((place.numel() - place.sum()) / pos).clamp(1.0, 60.0)
                loss_valid = F.binary_cross_entropy_with_logits(
                    valid_logit, place, pos_weight=pw)
                diff = F.smooth_l1_loss(dmg_pred, y_dmg.clamp(0, 1.2), reduction="none")
                # Breakable cells weigh like placeable ones: the dmg head is shared, and a
                # wrong break-field argmax is as bad as a wrong placement argmax.
                w = 1.0 + 3.0 * place + 3.0 * brk
                has_sup = (place + brk).flatten(1).sum(1) > 0
                mask = has_sup & trn_t["has_dmg"][idx]
                loss_dmg = (diff * w).flatten(1)[mask].mean() if mask.any() \
                    else torch.zeros(())
                loss = loss_valid + loss_dmg
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["loss"] += loss.item()
            tot["valid"] += loss_valid.item()
            tot["dmg"] += loss_dmg.item()
            nb += 1
            if bi % 10 == 0 or bi == n_batches - 1:
                print(f"  epoch {epoch} [{(bi + 1) * 100 // n_batches:>3}%] "
                      f"loss {loss.item():.4f} (valid {loss_valid.item():.3f} "
                      f"dmg {loss_dmg.item():.3f})", flush=True)
        sched.step()

        m = quick_eval(model, val_t, amp)
        rec = {"epoch": epoch, "loss": tot["loss"] / nb, "loss_valid": tot["valid"] / nb,
               "loss_dmg": tot["dmg"] / nb, "sec": time.time() - t0, **m}
        hist.append(rec)
        print(f"epoch {epoch}: loss {rec['loss']:.4f} | val top1 {m['top1']:.3f} "
              f"top5 {m['top5']:.3f} regret {m['regret']:.3f} | {rec['sec']:.0f}s",
              flush=True)
        if m["top1"] > best_top1:
            best_top1 = m["top1"]
            # CPU-side export: loads on any machine, GPU or not.
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "config": {"channels": CHANNELS, "aux_dim": ds["aux"].shape[1],
                                   "width": args.width, "factorized": args.factorized},
                        "epoch": epoch}, out / "model.pt")
        save_ckpt(epoch)

    (out / "log.json").write_text(json.dumps(hist, indent=1))
    print(f"best val top1: {best_top1:.3f} -> {out / 'model.pt'}", flush=True)


def quick_eval(model, val_t, amp=False):
    """Oracle-gated top-k recall + regret against the teacher's own best cell."""
    model.eval()
    tops1 = []
    tops5 = []
    regrets = []
    with torch.no_grad(), torch.autocast(val_t["x"].device.type,
                                         dtype=torch.bfloat16, enabled=amp):
        for i in range(0, len(val_t["x"]), 512):
            sl = slice(i, i + 512)
            dmg_pred, _ = model(val_t["x"][sl].float(), val_t["aux"][sl])
            place = val_t["place"][sl]
            dmg_lbl = val_t["dmg"][sl]
            self_dmg = val_t["self_dmg"][sl]
            s_health = val_t["s_health"][sl][:, None, None, None]
            gate = (place > 0) & (self_dmg < s_health - 1.0)
            for b in range(dmg_pred.shape[0]):
                g = gate[b]
                if not g.any():
                    continue
                p = torch.where(g, dmg_pred[b], torch.full_like(dmg_pred[b], -1e9))
                lbl = torch.where(g, dmg_lbl[b], torch.full_like(dmg_lbl[b], -1e9))
                best_idx = int(lbl.flatten().argmax())
                top5 = torch.topk(p.flatten(), 5).indices.tolist()
                tops1.append(1.0 if best_idx in top5[:1] else 0.0)
                tops5.append(1.0 if best_idx in top5 else 0.0)
                regrets.append(float(dmg_lbl[b].flatten()[best_idx]
                                     - dmg_lbl[b].flatten()[top5[0]]))
    return {"top1": float(np.mean(tops1)), "top5": float(np.mean(tops5)),
            "regret": float(np.mean(regrets)),
            "n": len(tops1)}


if __name__ == "__main__":
    main()
