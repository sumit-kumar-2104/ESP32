"""
Task-A A/B runner — short in-domain train on dfs_full so you can compare:

    baseline  : continuous ComplexLinear, no R1/R2 hooks (Task-A control)
    R1        : ComplexLinear + weight-decay + input dropout + optional
                complex-dim bottleneck + early-stop on val (fix continuous
                overfit — train 100% / val 41%)
    R2        : DiscreteComplexLinear + grad-scaled STE (--qgrad-scale) +
                optional separate LR (--lr-complex) + optional hardtanh
                surrogate (fix quantized grad-starvation — grad ~14x smaller
                than plain-linear control)

Task-A diagnosed:
    - RAW CSI probe ceiling ~95% >> DFS probe ceiling ~63%. DFS OTA
      target is honest ~50-60%, NOT 80%.
    - STE WORKS on the current DiscreteComplexLinear (grad flows). Both
      failures are magnitude / capacity issues, not a dead gradient.
    - Continuous ComplexLinear OVERFITS on dfs_full (train 100 / val 41)
      vs plain-linear control (val 62).
    - 2-bit quantized UNDER-LEARNS: grad thru complex-linear ~0.08 vs
      control ~1.13 (~14x smaller), softmax entropy stuck near uniform.

This script does NOT touch STE as a "fix" — HOOK STE / HOOK_HEAD_LR are
independent of R1/R2 and stay disabled here. R1 / R2 are opt-in via
`--enable {R1,R2,none}`; without `--enable`, baseline runs.

Hard rules honoured:
    - prints device; timestamped log to logs/; seed 42 by default
    - never fabricates data (fails loudly on missing raw CSI, via
      `require_raw_csi_dir()`)
    - new file, new opt-in flags — does not modify working scripts

Usage examples (server):
    # baseline
    python train_ab_dfs.py --dates 20181109 --epochs 40 \
        | tee logs/dfs_ab_baseline_$(date +%Y%m%d_%H%M%S).log

    # R1: continuous overfit fix
    python train_ab_dfs.py --dates 20181109 --epochs 40 \
        --enable R1 --wd 1e-3 --dropout 0.4 --complex-dim 512 --patience 6 \
        | tee logs/dfs_ab_R1_$(date +%Y%m%d_%H%M%S).log

    # R2: quantized under-learn fix
    python train_ab_dfs.py --dates 20181109 --epochs 40 \
        --enable R2 --qgrad-scale 8.0 --lr-complex 5e-3 --ste hardtanh \
        | tee logs/dfs_ab_R2_$(date +%Y%m%d_%H%M%S).log
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, print_device, set_seed
from data.csi_loader import build_csi_features
from models.linear_complex import ComplexLinear
from models.discrete_nn import DiscreteComplexLinear
import fix_pipeline as fp


# ─── Data (same conventions as diagnose_training.py) ─────────────────────────

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


# ─── Model factories ─────────────────────────────────────────────────────────

def build_model(which: str, in_dim: int, num_classes: int) -> nn.Module:
    """Return the model that goes with the A/B arm.

    which == 'baseline' -> continuous ComplexLinear, R1/R2 OFF (asserted)
    which == 'R1'       -> R1-wrapped continuous ComplexLinear
    which == 'R2'       -> R2-wrapped DiscreteComplexLinear (2-bit)
    """
    if which == "baseline":
        assert not fp.is_hook_R1_enabled() and not fp.is_hook_R2_enabled(), (
            "baseline arm expects both R1 and R2 disabled — got "
            f"R1={fp.is_hook_R1_enabled()}  R2={fp.is_hook_R2_enabled()}"
        )
        return ComplexLinear(in_dim, num_classes)
    if which == "R1":
        assert fp.is_hook_R1_enabled(), "R1 arm requires enable_hook_R1() called"
        return fp.make_r1_ota_model(in_dim, num_classes)
    if which == "R2":
        assert fp.is_hook_R2_enabled(), "R2 arm requires enable_hook_R2() called"
        return fp.make_r2_ota_model(in_dim, num_classes)
    raise ValueError(f"unknown arm {which!r}")


def build_optimizer(which: str, model: nn.Module, lr: float, wd: float):
    if which == "R2":
        groups = fp.r2_param_groups(model, base_lr=lr)
        # weight_decay applied uniformly (small); R2 doesn't override it
        return torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _softmax_entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1).clamp_min(1e-12)
    return float(-(p * p.log()).sum(dim=-1).mean().item())


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


# ─── Training loop with instrumentation + early-stop ─────────────────────────

def train_arm(which: str, X_tr, y_tr, X_te, y_te, in_dim, num_classes,
              device, args) -> dict:
    torch.manual_seed(args.seed)
    model = build_model(which, in_dim, num_classes).to(device)
    wd = fp.r1_config()["wd"] if which == "R1" else args.wd
    opt = build_optimizer(which, model, lr=args.lr, wd=wd)
    ce = nn.CrossEntropyLoss()

    xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    xte = torch.tensor(X_te, dtype=torch.float32, device=device)
    yte = torch.tensor(y_te, dtype=torch.long, device=device)
    n = xtr.shape[0]

    # Track weight_real/weight_imag whether they live directly on `model`
    # (ComplexLinear / DiscreteComplexLinear) or on `model.complex` (R1).
    def _complex_params():
        if hasattr(model, "complex"):
            return [model.complex.weight_real, model.complex.weight_imag]
        return [model.weight_real, model.weight_imag]

    history = []
    best_val = -1.0
    best_ep = 0
    stale = 0
    patience = fp.r1_config()["patience"] if which == "R1" else args.patience

    print(f"\n  ARM = {which}   patience = {patience}   wd = {wd}")
    print(f"  {'ep':>3} {'loss':>10} {'train':>7} {'val':>7} "
          f"{'|g(W)|':>10} {'H(sm)':>8} {'|y|mean':>10}")

    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        model.train()
        tot_loss = n_correct = n_seen = 0
        g_W_sum = 0.0
        n_steps = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            if which in ("baseline", "R2"):
                xc = torch.complex(xb, torch.zeros_like(xb))
                logits = model(xc)
            else:  # R1 wrapper accepts real input directly
                logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            g_W_sum += _grad_norm(_complex_params())
            opt.step()
            tot_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
            n_correct += int((logits.argmax(1) == yb).sum().item())
            n_steps += 1
        train_loss = tot_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        avg_gW = g_W_sum / max(n_steps, 1)

        model.eval()
        with torch.no_grad():
            if which in ("baseline", "R2"):
                xc = torch.complex(xte, torch.zeros_like(xte))
                logits = model(xc)
            else:
                logits = model(xte)
            val_acc = float((logits.argmax(1) == yte).float().mean().item())
            H_sm = _softmax_entropy(logits)
            y_mag_mean = float(logits.abs().mean().item())

        history.append({
            "epoch": ep, "loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "g_W": avg_gW,
            "sm_entropy": H_sm, "y_mag_mean": y_mag_mean,
        })
        print(f"  {ep:>3d} {train_loss:>10.4f} {train_acc*100:>6.1f}% "
              f"{val_acc*100:>6.1f}% {avg_gW:>10.3g} "
              f"{H_sm:>8.3f} {y_mag_mean:>10.3g}")

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_ep = ep
            stale = 0
        else:
            stale += 1
        if patience is not None and patience > 0 and stale >= patience:
            print(f"  [early-stop] no val improvement for {patience} epochs "
                  f"(best {best_val*100:.2f}% at ep {best_ep}).")
            break

    return {
        "history": history,
        "best_val": best_val,
        "best_ep": best_ep,
        "final_val": history[-1]["val_acc"],
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", choices=["dfs_full", "dfs_small"],
                    default="dfs_full")
    ap.add_argument("--dates", nargs="+", default=["20181109"])
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csi-root", default=None)

    # ── A/B selector
    ap.add_argument("--enable", choices=["none", "R1", "R2"], default="none",
                    help="which fix hook to enable for THIS run "
                         "(the runner does ONE arm per invocation, so the "
                         "server can `tee` each to its own log). "
                         "default 'none' = Task-A baseline.")

    # ── R1 knobs (all opt-in, ignored when --enable != R1)
    ap.add_argument("--wd", type=float, default=1e-4,
                    help="weight_decay (R1 or baseline). Task-A default 1e-4.")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="input dropout probability (R1).")
    ap.add_argument("--complex-dim", type=int, default=None,
                    help="optional real-linear bottleneck to shrink the input "
                         "dim BEFORE the ComplexLinear (R1).")
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stop patience on val_acc (R1 / baseline).")

    # ── R2 knobs (all opt-in, ignored when --enable != R2)
    ap.add_argument("--qgrad-scale", type=float, default=8.0,
                    help="multiply STE backward grad by this factor to offset "
                         "the observed ~14x quantized-grad shrink (R2).")
    ap.add_argument("--lr-complex", type=float, default=None,
                    help="separate LR for weight_real/weight_imag (R2). "
                         "If unset, all params use --lr.")
    ap.add_argument("--ste", choices=["identity", "hardtanh"],
                    default="identity",
                    help="STE surrogate. 'identity' = current STE with grad "
                         "scaling only. 'hardtanh' = clip grad to [-1,1] "
                         "before scaling (R2).")

    args = ap.parse_args()

    # Reset any leftover hook state, then opt in to the requested arm.
    fp.disable_all_hooks()
    if args.enable == "R1":
        fp.enable_hook_R1(wd=args.wd, dropout=args.dropout,
                          complex_dim=args.complex_dim, patience=args.patience)
    elif args.enable == "R2":
        fp.enable_hook_R2(qgrad_scale=args.qgrad_scale,
                          lr_complex=args.lr_complex, ste_kind=args.ste)

    log_path = setup_logging(f"dfs_ab_{args.enable.lower()}")
    print(f"[ab] Task-A A/B — arm={args.enable}   features={args.features}")
    print(f"[ab] log = {log_path}   ts = {time.strftime('%Y%m%d_%H%M%S')}")
    device = print_device()
    set_seed(args.seed)
    if args.enable == "R1":
        print(f"[ab] R1 cfg = {fp.r1_config()}")
    elif args.enable == "R2":
        print(f"[ab] R2 cfg = {fp.r2_config()}")

    X, y = load_dfs(args)
    n_classes = int(np.unique(y).size)
    print(f"[data] N={len(X)}  dim={X.shape[1]}  classes={n_classes}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )
    sc = StandardScaler().fit(X_tr)
    # Sanity: StandardScaler must be fit on the TRAIN fold only.
    assert getattr(sc, "n_samples_seen_", 0) == len(X_tr), (
        "StandardScaler was NOT fit only on the training fold "
        f"(n_samples_seen_={getattr(sc, 'n_samples_seen_', None)} != "
        f"len(X_tr)={len(X_tr)}). Refusing to train on leaked stats.")
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    print(f"[data] train={len(X_tr_s)}  test={len(X_te_s)}  "
          f"train std={X_tr_s.std():.3g}  train mean={X_tr_s.mean():.3g}")

    which = "baseline" if args.enable == "none" else args.enable
    result = train_arm(which, X_tr_s, y_tr, X_te_s, y_te,
                       X_tr_s.shape[1], n_classes, device, args)

    print("\n" + "=" * 70)
    print(f"  RESULT — arm={which}")
    print("=" * 70)
    print(f"  best val acc  = {result['best_val']*100:.2f}%  (epoch {result['best_ep']})")
    print(f"  final val acc = {result['final_val']*100:.2f}%")
    print(f"  DFS probe ceiling ~63% (MLP) / ~61% (LogReg) — Task-A honest floor.")
    print(f"[done] log saved to {log_path}")


if __name__ == "__main__":
    main()
