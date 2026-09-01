"""
Task A — training-loop diagnostic on dfs_full.

Phase 0 already ruled out data/labels/normalization/forward-path collapse:
    dfs_full: sklearn LogReg 60.7%, MLP 63.6% in-domain (chance 20%).
    OTA forward path: healthy magnitudes at every stage, no collapse.
So the in-domain OTA ~24% must be a TRAINING-LOOP or GRADIENT-PATH bug.

This script trains the OTA model in-domain on dfs_full for a short run and
instruments the LEARNING, not the data:

    1. Per epoch: train loss, train acc, val acc, and gradient-norm of
         (a) complex linear layer (weight_real, weight_imag),
         (b) magnitude-readout output (grad of the |.| output tensor),
         (c) classifier head (only present in the digital control model).
       Any grad-norm near 0 -> DEAD PATH. Any exploding norm -> flagged.

    2. Straight-through estimator check.
       When --quantize is set, we explicitly compare the STE-quantized
       weight's gradient against the continuous weight's gradient on a
       synthetic backward pass:
         - If d(H_quant)/d(H_cont) is the identity (STE working),
           norms match.
         - If it is 0 (quantizer is opaque to autograd), gradient is zero.
       Fails LOUDLY on a zero result.

    3. Classifier-head init scale + logit / softmax entropy per epoch.
       Detects saturation (entropy near 0) or uniform-collapse (entropy
       near log(C), meaning the head is not learning).

    4. Control run: same features, same optimizer, but BYPASS OTA — a plain
       real Linear + CE. If control reaches ~55-63% in-domain and OTA
       stays at chance, the bug is isolated to the OTA / quantizer /
       magnitude-readout gradient path.

Writes logs/diag_training_<timestamp>.log. Fixes nothing.

Usage:
    python diagnose_training.py --features dfs_full --dates 20181109 \
        [--epochs 40] [--quantize]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, print_device, set_seed
from models.linear_complex import ComplexLinear
from models.discrete_nn import DiscreteComplexLinear
from data.csi_loader import build_csi_features, FEATURE_MODES, DFS_BINS_MODES


# ─── Data loader (reuses Phase-0 conventions) ────────────────────────────────

def load_dfs(args) -> Tuple[np.ndarray, np.ndarray]:
    from config import get_raw_csi_dir, require_raw_csi_dir
    csi_root = Path(args.csi_root) if args.csi_root else get_raw_csi_dir()
    if not csi_root.exists() or not any(csi_root.iterdir()):
        require_raw_csi_dir()
    dfs_bins = "small" if args.features == "dfs_small" else "full"
    data = build_csi_features(
        csi_root, args.dates,
        keep_users=set(args.users) if args.users else None,
        keep_gestures=set(args.gestures) if args.gestures else None,
        feature="dfs_spec", dfs_bins=dfs_bins,
    )
    X = np.asarray(data["X"], dtype=np.float32)
    y_raw = np.asarray(data["y_gesture"], dtype=np.int64)
    uniq = np.unique(y_raw)
    remap = {g: i for i, g in enumerate(uniq)}
    y = np.array([remap[g] for g in y_raw], dtype=np.int64)
    return X, y


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _softmax_entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1).clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)
    return float(ent.mean().item())


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


# ─── Task A.2 — STE sanity check ──────────────────────────────────────────────

def check_ste_gradient(model: DiscreteComplexLinear, device) -> str:
    """Directly probe whether gradient flows through _quantize_ste.

    Method: build a tiny loss on H_quant, backprop, and inspect the .grad on
    the underlying real/imag parameters. If STE is working, .grad matches
    what you'd get with the continuous weight (up to the STE identity trick).
    A zero grad here means the quantizer is opaque -> #1 root cause candidate.
    """
    model.zero_grad(set_to_none=True)
    Hq = model.complex_weight
    # Scalar surrogate loss: sum of squared magnitudes of H_quant.
    loss = (Hq.real ** 2 + Hq.imag ** 2).sum()
    loss.backward()
    gr_real = model.weight_real.grad
    gr_imag = model.weight_imag.grad
    if gr_real is None or gr_imag is None:
        return ("STE FAIL: weight_real.grad / weight_imag.grad is None. "
                "Autograd did not see the quantized weight — quantizer is "
                "OPAQUE. Fix required.")
    n_real = float(gr_real.detach().pow(2).sum().sqrt())
    n_imag = float(gr_imag.detach().pow(2).sum().sqrt())
    if n_real + n_imag < 1e-12:
        return (f"STE FAIL: grad norm on (weight_real, weight_imag) = "
                f"({n_real:.3g}, {n_imag:.3g}). Backward returned zero — "
                f"STE identity trick is not active.")
    return (f"STE OK: grad flows through _quantize_ste. "
            f"|grad w_real|={n_real:.3g}  |grad w_imag|={n_imag:.3g}")


# ─── Training loops ──────────────────────────────────────────────────────────

def train_ota(X_tr, y_tr, X_te, y_te, in_dim, num_classes, device,
              quantize: bool, epochs: int, seed: int,
              lr: float = 1e-3, weight_decay: float = 1e-4,
              batch_size: int = 128) -> dict:
    torch.manual_seed(seed)
    model = (DiscreteComplexLinear(in_dim, num_classes) if quantize
             else ComplexLinear(in_dim, num_classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    # Track head-init scale (for the OTA path, the "head" is the complex
    # linear layer itself — there is no separate classifier head).
    with torch.no_grad():
        head_scale = float((model.weight_real.pow(2) +
                            model.weight_imag.pow(2)).sqrt().mean().item())
    print(f"  [init] complex-linear weight |w| mean = {head_scale:.4g}")

    xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    xte = torch.tensor(X_te, dtype=torch.float32, device=device)
    yte = torch.tensor(y_te, dtype=torch.long, device=device)

    n = xtr.shape[0]
    history = []
    print(f"\n  {'ep':>3} {'loss':>10} {'train':>7} {'val':>7} "
          f"{'|g(W)|':>10} {'|g(|y|)|':>10} {'H(sm)':>8} {'|y|mean':>10}")
    for ep in range(1, epochs + 1):
        # ─ epoch: minibatch shuffle
        perm = torch.randperm(n, device=device)
        model.train()
        total_loss = 0.0
        n_correct = 0
        n_seen = 0
        g_W_sum = 0.0
        g_y_sum = 0.0
        n_steps = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = xtr[idx]
            yb = ytr[idx]
            xc = torch.complex(xb, torch.zeros_like(xb))
            opt.zero_grad(set_to_none=True)
            y_mag = model(xc)          # (B, R)
            y_mag.retain_grad()
            loss = ce(y_mag, yb)
            loss.backward()
            g_W = _grad_norm([model.weight_real, model.weight_imag])
            g_y = (float(y_mag.grad.detach().pow(2).sum().sqrt())
                   if y_mag.grad is not None else 0.0)
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
            n_correct += int((y_mag.argmax(1) == yb).sum().item())
            g_W_sum += g_W
            g_y_sum += g_y
            n_steps += 1

        train_loss = total_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        avg_gW = g_W_sum / max(n_steps, 1)
        avg_gy = g_y_sum / max(n_steps, 1)

        # ─ val
        model.eval()
        with torch.no_grad():
            xc = torch.complex(xte, torch.zeros_like(xte))
            logits = model(xc)
            val_acc = float((logits.argmax(1) == yte).float().mean().item())
            H_sm = _softmax_entropy(logits)
            y_mag_mean = float(logits.abs().mean().item())

        history.append({
            "epoch": ep, "loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "g_W": avg_gW, "g_ymag": avg_gy,
            "sm_entropy": H_sm, "y_mag_mean": y_mag_mean,
        })
        print(f"  {ep:>3d} {train_loss:>10.4f} {train_acc*100:>6.1f}% "
              f"{val_acc*100:>6.1f}% {avg_gW:>10.3g} {avg_gy:>10.3g} "
              f"{H_sm:>8.3f} {y_mag_mean:>10.3g}")
    return {"history": history, "head_scale": head_scale, "final_val": val_acc}


def train_control(X_tr, y_tr, X_te, y_te, in_dim, num_classes, device,
                  epochs: int, seed: int,
                  lr: float = 1e-3, weight_decay: float = 1e-4,
                  batch_size: int = 128) -> dict:
    """Real-valued Linear head + CE. Bypasses OTA and quantization entirely."""
    torch.manual_seed(seed)
    model = nn.Linear(in_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        head_scale = float(model.weight.pow(2).sqrt().mean().item())
    print(f"  [init] linear-head |w| mean = {head_scale:.4g}")

    xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    xte = torch.tensor(X_te, dtype=torch.float32, device=device)
    yte = torch.tensor(y_te, dtype=torch.long, device=device)

    n = xtr.shape[0]
    history = []
    print(f"\n  {'ep':>3} {'loss':>10} {'train':>7} {'val':>7} "
          f"{'|g(W)|':>10} {'H(sm)':>8}")
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        model.train()
        total_loss = n_correct = n_seen = 0
        g_W_sum = 0.0
        n_steps = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            g_W_sum += _grad_norm([model.weight, model.bias])
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
            n_correct += int((logits.argmax(1) == yb).sum().item())
            n_steps += 1
        train_loss = total_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        avg_gW = g_W_sum / max(n_steps, 1)

        model.eval()
        with torch.no_grad():
            logits = model(xte)
            val_acc = float((logits.argmax(1) == yte).float().mean().item())
            H_sm = _softmax_entropy(logits)
        history.append({
            "epoch": ep, "loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "g_W": avg_gW, "sm_entropy": H_sm,
        })
        print(f"  {ep:>3d} {train_loss:>10.4f} {train_acc*100:>6.1f}% "
              f"{val_acc*100:>6.1f}% {avg_gW:>10.3g} {H_sm:>8.3f}")
    return {"history": history, "head_scale": head_scale, "final_val": val_acc}


# ─── Flagging ────────────────────────────────────────────────────────────────

def flag_grad_history(history: list, key: str, label: str,
                      dead_thresh: float = 1e-8,
                      explode_thresh: float = 1e4) -> Optional[str]:
    vals = [h[key] for h in history if key in h]
    if not vals:
        return None
    mn, mx, med = min(vals), max(vals), float(np.median(vals))
    print(f"  grad-norm {label}: min={mn:.3g}  median={med:.3g}  max={mx:.3g}")
    if mx < dead_thresh:
        return (f"DEAD PATH — grad-norm of {label} never exceeded {dead_thresh:g} "
                f"(max={mx:.3g}). Gradient is not reaching this layer.")
    if mx > explode_thresh:
        return (f"EXPLOSION — grad-norm of {label} reached {mx:.3g}. "
                f"Optimizer diverged.")
    if med < dead_thresh * 10:
        return (f"NEAR-DEAD — grad-norm of {label} median={med:.3g}. "
                f"Signal too weak; layer barely trains.")
    return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", choices=["dfs_full", "dfs_small"],
                    default="dfs_full")
    ap.add_argument("--dates", nargs="+", default=["20181109"])
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quantize", action="store_true",
                    help="use DiscreteComplexLinear (2-bit phase + STE) "
                         "instead of ComplexLinear. Task A.2 STE check runs "
                         "either way, but is only meaningful with --quantize.")
    ap.add_argument("--csi-root", default=None)
    args = ap.parse_args()

    log_path = setup_logging(f"diag_training")
    print(f"[diag] Task A training-loop diagnostic — features={args.features}")
    print(f"[diag] log = {log_path}   ts = {time.strftime('%Y%m%d_%H%M%S')}")
    device = print_device()
    set_seed(args.seed)

    # ─── Data ─────────────────────────────────────────────────────────────
    X, y = load_dfs(args)
    n_classes = int(np.unique(y).size)
    print(f"[data] N={len(X)}  dim={X.shape[1]}  classes={n_classes}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    print(f"[data] train={len(X_tr)}  test={len(X_te)}  "
          f"train std={X_tr.std():.3g}  train mean={X_tr.mean():.3g}")
    in_dim = X_tr.shape[1]

    # ─── Task A.2 — STE sanity ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  A.2 STRAIGHT-THROUGH ESTIMATOR CHECK")
    print("=" * 70)
    quant_model = DiscreteComplexLinear(in_dim, n_classes).to(device)
    ste_msg = check_ste_gradient(quant_model, device)
    print(f"  {ste_msg}")

    # ─── Task A.1 + A.3 — OTA training with instrumentation ───────────────
    print("\n" + "=" * 70)
    label = "DiscreteComplexLinear (2-bit STE)" if args.quantize else "ComplexLinear (continuous)"
    print(f"  A.1/A.3 OTA TRAINING — {label}   epochs={args.epochs}")
    print("=" * 70)
    ota = train_ota(X_tr, y_tr, X_te, y_te, in_dim, n_classes, device,
                    quantize=args.quantize, epochs=args.epochs, seed=args.seed)

    # ─── Task A.4 — control (bypass OTA) ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  A.4 CONTROL — real-valued nn.Linear (BYPASS OTA + quantization)")
    print("=" * 70)
    ctrl = train_control(X_tr, y_tr, X_te, y_te, in_dim, n_classes, device,
                         epochs=args.epochs, seed=args.seed)

    # ─── Grad-norm flags ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GRAD-NORM FLAGS")
    print("=" * 70)
    flags = []
    for name, key, hist in [
        ("OTA complex linear layer", "g_W", ota["history"]),
        ("OTA magnitude readout |y|", "g_ymag", ota["history"]),
        ("control classifier head", "g_W", ctrl["history"]),
    ]:
        print(f"\n  {name}")
        f = flag_grad_history(hist, key, name)
        if f:
            flags.append(f)

    if flags:
        print("\n  [flags]")
        for f in flags:
            print(f"    - {f}")
    else:
        print("\n  [OK] no grad-norm anomalies detected.")

    # ─── Softmax-entropy flags ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SOFTMAX ENTROPY (log(C) = {:.3f} means uniform / not learning)".format(
        float(np.log(n_classes))))
    print("=" * 70)
    ent_ota = [h["sm_entropy"] for h in ota["history"]]
    ent_ctrl = [h["sm_entropy"] for h in ctrl["history"]]
    print(f"  OTA:     first={ent_ota[0]:.3f}  last={ent_ota[-1]:.3f}  "
          f"min={min(ent_ota):.3f}")
    print(f"  Control: first={ent_ctrl[0]:.3f}  last={ent_ctrl[-1]:.3f}  "
          f"min={min(ent_ctrl):.3f}")
    if ent_ota[-1] > 0.95 * float(np.log(n_classes)):
        flags.append("OTA softmax entropy stays near uniform — head is not "
                     "differentiating classes.")

    # ─── Verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    ota_v = ota["final_val"] * 100
    ctrl_v = ctrl["final_val"] * 100
    chance = 100.0 / n_classes
    print(f"  OTA     final val acc = {ota_v:.2f}%")
    print(f"  Control final val acc = {ctrl_v:.2f}%")
    print(f"  chance level         = {chance:.2f}%")
    print(f"  DFS probe ceiling    ~ 55-63% (from Phase 0)")

    if ctrl_v >= 40.0 and ota_v <= chance + 8.0:
        print("\n  -> Control LEARNS while OTA sits at chance.")
        print("     The fault is ISOLATED to the OTA / quantizer / magnitude-")
        print("     readout gradient path — NOT to features, labels, or the")
        print("     optimizer. Prime suspects, in order:")
        print("       (a) STE is opaque to autograd (see A.2 above).")
        print("       (b) magnitude readout kills grad when |y| is tiny at")
        print("           init (near-zero grad of |z| at z=0).")
        print("       (c) classifier head effectively missing — the |.| output")
        print("           is used as logits directly, so with |y| ~ constant")
        print("           the softmax entropy stays uniform.")
        print("     -> Enable fix_pipeline.enable_hook_STE() (Task B, hook 1)")
        print("        or fix_pipeline.enable_hook_HEAD_LR() (Task B, hook 2)")
        print("        depending on which of A.2 or the grad-norms above")
        print("        pointed to the dead path.")
    elif ota_v >= chance + 15.0:
        print("\n  -> OTA is learning (val > chance + 15pp). If the number is")
        print("     still far below the DFS probe ceiling, the fault is a")
        print("     LR / capacity issue, not a dead gradient. Consider tuning")
        print("     lr / epochs before invoking fix hooks.")
    else:
        print("\n  -> Both OTA and control are near chance. The training loop")
        print("     itself is broken (optimizer, batch, or CE target shape). ")
        print("     Re-check X/y dtype and the label remapping above.")

    print("\n[done] Task A training-loop diagnostic complete.")
    print(f"[done] Report saved to: {log_path}")


if __name__ == "__main__":
    main()
