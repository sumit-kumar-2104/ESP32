"""
Task-B raw-CSI Phase-2 in-domain trainer (OTA model).

Phase 0 measured RAW CSI probe ceiling ~95% (LogReg 92.67% / MLP 94.67%) vs
DFS ~63%. This script is the end-to-end trainer that turns raw CSI into an
actual OTA in-domain number, gated by `--target-acc` (default 0.70 for raw
CSI, honest DFS is 0.50 and lives in `train_ab_dfs.py`).

Guarantees:
    * If the raw Widar3.0 CSI directory is missing OR empty, the script
      calls `require_raw_csi_dir()` which prints the exact IEEE-DataPort
      download instructions and exits — NEVER silently falls back to DFS.
    * StandardScaler is fit on the TRAIN fold only. This is asserted at
      runtime (`sc.n_samples_seen_ == len(X_train)`).
    * `--target-acc` gates the run via `assert_indomain_ok` — raw CSI must
      clear the target or the script fails LOUDLY (no misleading downstream
      cross-room numbers).
    * The R1 / R2 fix hooks are AVAILABLE via the same `--enable {R1,R2}`
      flag as `train_ab_dfs.py`, so the raw-CSI run can reuse whichever
      Task-A A/B winner works best. Baseline is 'none'.

Hard rules honoured:
    - prints device; timestamped log to logs/; seed 42 by default
    - new file, new opt-in flags; existing scripts unchanged
    - never fabricates numbers (loud assert on missing data)

Usage examples (server):
    # Baseline raw-CSI in-domain
    python train_raw_csi.py --dates 20181109 --epochs 40 --target-acc 0.70 \
        | tee logs/raw_csi_baseline_$(date +%Y%m%d_%H%M%S).log

    # With R1 (continuous overfit fix)
    python train_raw_csi.py --dates 20181109 --epochs 40 --target-acc 0.70 \
        --enable R1 --wd 1e-3 --dropout 0.4 --complex-dim 1024 --patience 6 \
        | tee logs/raw_csi_R1_$(date +%Y%m%d_%H%M%S).log

    # With R2 (2-bit quantized OTA)
    python train_raw_csi.py --dates 20181109 --epochs 40 --target-acc 0.70 \
        --enable R2 --qgrad-scale 8.0 --lr-complex 5e-3 --ste hardtanh \
        | tee logs/raw_csi_R2_$(date +%Y%m%d_%H%M%S).log
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
import fix_pipeline as fp
from fix_pipeline import assert_indomain_ok


# ─── Data ────────────────────────────────────────────────────────────────────

def load_raw_csi(args) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw Widar3.0 CSI as the 5760-dim per-sample feature.

    Fails LOUDLY (via `require_raw_csi_dir()`) if the raw CSI directory is
    missing. There is NO silent DFS fallback.
    """
    from config import get_raw_csi_dir, require_raw_csi_dir
    csi_root = Path(args.csi_root) if args.csi_root else get_raw_csi_dir()
    if not csi_root.exists() or not any(csi_root.iterdir()):
        require_raw_csi_dir()   # prints download+target-path msg and sys.exit(1)

    # Sanity: even if the root exists it may be empty of .dat files. Build a
    # tiny probe list to fail with the same message when nothing loads.
    data = build_csi_features(
        csi_root, args.dates,
        keep_users=set(args.users) if args.users else None,
        keep_gestures=set(args.gestures) if args.gestures else None,
        feature="raw", dfs_bins="full",
    )
    X = np.asarray(data["X"], dtype=np.float32)
    y_raw = np.asarray(data["y_gesture"], dtype=np.int64)
    if len(X) == 0:
        require_raw_csi_dir()
    uniq = np.unique(y_raw)
    remap = {g: i for i, g in enumerate(uniq)}
    y = np.array([remap[g] for g in y_raw], dtype=np.int64)
    return X, y


# ─── Model / optimizer factories (same conventions as train_ab_dfs.py) ────────

def build_model(which: str, in_dim: int, num_classes: int) -> nn.Module:
    if which == "baseline":
        assert not fp.is_hook_R1_enabled() and not fp.is_hook_R2_enabled()
        return ComplexLinear(in_dim, num_classes)
    if which == "R1":
        assert fp.is_hook_R1_enabled()
        return fp.make_r1_ota_model(in_dim, num_classes)
    if which == "R2":
        assert fp.is_hook_R2_enabled()
        return fp.make_r2_ota_model(in_dim, num_classes)
    raise ValueError(f"unknown arm {which!r}")


def build_optimizer(which: str, model: nn.Module, lr: float, wd: float):
    if which == "R2":
        return torch.optim.Adam(fp.r2_param_groups(model, base_lr=lr),
                                lr=lr, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _softmax_entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1).clamp_min(1e-12)
    return float(-(p * p.log()).sum(dim=-1).mean().item())


# ─── Training loop ──────────────────────────────────────────────────────────

def train_indomain(which: str, X_tr, y_tr, X_te, y_te, in_dim, num_classes,
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

    patience = fp.r1_config()["patience"] if which == "R1" else args.patience
    print(f"\n  ARM = {which}   patience = {patience}   wd = {wd}")
    print(f"  {'ep':>3} {'loss':>10} {'train':>7} {'val':>7} "
          f"{'H(sm)':>8} {'|y|mean':>10}")

    history = []
    best_val = -1.0
    best_ep = 0
    stale = 0
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        model.train()
        tot_loss = n_correct = n_seen = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            if which in ("baseline", "R2"):
                logits = model(torch.complex(xb, torch.zeros_like(xb)))
            else:
                logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
            n_correct += int((logits.argmax(1) == yb).sum().item())
        train_loss = tot_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            if which in ("baseline", "R2"):
                logits = model(torch.complex(xte, torch.zeros_like(xte)))
            else:
                logits = model(xte)
            val_acc = float((logits.argmax(1) == yte).float().mean().item())
            H_sm = _softmax_entropy(logits)
            y_mag_mean = float(logits.abs().mean().item())

        history.append({
            "epoch": ep, "loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "sm_entropy": H_sm, "y_mag_mean": y_mag_mean,
        })
        print(f"  {ep:>3d} {train_loss:>10.4f} {train_acc*100:>6.1f}% "
              f"{val_acc*100:>6.1f}% {H_sm:>8.3f} {y_mag_mean:>10.3g}")

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

    return {"history": history, "best_val": best_val, "best_ep": best_ep,
            "final_val": history[-1]["val_acc"]}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", choices=["raw_csi"], default="raw_csi",
                    help="Only 'raw_csi' is supported by this script; no "
                         "silent fallback to DFS. Present as an explicit flag "
                         "so the command matches Phase-2 docs.")
    ap.add_argument("--dates", nargs="+", default=["20181109"])
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csi-root", default=None,
                    help="override the raw Widar3.0 CSI directory. If unset, "
                         "resolves via METAAI_RAW_CSI_DIR / METAAI_DATA_DIR "
                         "in config.get_raw_csi_dir().")

    # Task-C gate
    ap.add_argument("--target-acc", "--indomain-threshold", type=float,
                    default=0.70, dest="target_acc",
                    help="in-domain acc floor gated via assert_indomain_ok. "
                         "raw CSI default 0.70 (Phase-2 Task-B target). Pass a "
                         "smaller value ONLY for debugging.")

    # A/B arm (shared surface with train_ab_dfs.py)
    ap.add_argument("--enable", choices=["none", "R1", "R2"], default="none")
    # R1 knobs
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--complex-dim", type=int, default=None)
    ap.add_argument("--patience", type=int, default=8)
    # R2 knobs
    ap.add_argument("--qgrad-scale", type=float, default=8.0)
    ap.add_argument("--lr-complex", type=float, default=None)
    ap.add_argument("--ste", choices=["identity", "hardtanh"], default="identity")

    args = ap.parse_args()

    fp.disable_all_hooks()
    if args.enable == "R1":
        fp.enable_hook_R1(wd=args.wd, dropout=args.dropout,
                          complex_dim=args.complex_dim, patience=args.patience)
    elif args.enable == "R2":
        fp.enable_hook_R2(qgrad_scale=args.qgrad_scale,
                          lr_complex=args.lr_complex, ste_kind=args.ste)

    log_path = setup_logging(f"raw_csi_{args.enable.lower()}")
    print(f"[raw-csi] Task-B Phase-2 in-domain trainer — arm={args.enable}")
    print(f"[raw-csi] log = {log_path}   ts = {time.strftime('%Y%m%d_%H%M%S')}")
    device = print_device()
    set_seed(args.seed)
    if args.enable == "R1":
        print(f"[raw-csi] R1 cfg = {fp.r1_config()}")
    elif args.enable == "R2":
        print(f"[raw-csi] R2 cfg = {fp.r2_config()}")

    X, y = load_raw_csi(args)
    n_classes = int(np.unique(y).size)
    print(f"[data] N={len(X)}  dim={X.shape[1]}  classes={n_classes}")
    # Sanity: raw CSI is non-normalized. Expected mean ~6-8, std ~1 (Phase-0).
    print(f"[data] raw X: mean={X.mean():.3g}  std={X.std():.3g}  "
          f"min={X.min():.3g}  max={X.max():.3g}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )
    sc = StandardScaler().fit(X_tr)
    assert getattr(sc, "n_samples_seen_", 0) == len(X_tr), (
        "StandardScaler was NOT fit on TRAIN-fold-only "
        f"(n_samples_seen_={getattr(sc, 'n_samples_seen_', None)} != "
        f"len(X_tr)={len(X_tr)}).")
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    print(f"[data] train={len(X_tr_s)}  test={len(X_te_s)}  "
          f"post-scaler train std={X_tr_s.std():.3g}  mean={X_tr_s.mean():.3g}")

    which = "baseline" if args.enable == "none" else args.enable
    result = train_indomain(which, X_tr_s, y_tr, X_te_s, y_te,
                            X_tr_s.shape[1], n_classes, device, args)

    print("\n" + "=" * 70)
    print(f"  RESULT — arm={which}")
    print("=" * 70)
    print(f"  best val acc  = {result['best_val']*100:.2f}%  "
          f"(epoch {result['best_ep']})")
    print(f"  final val acc = {result['final_val']*100:.2f}%")
    print(f"  raw-CSI probe ceiling ~95% (Phase-0). Target-acc = "
          f"{args.target_acc*100:.0f}%.")

    # Loud gate — fails if val < target_acc.
    assert_indomain_ok(result["best_val"], target_acc=args.target_acc,
                       label="raw-CSI OTA in-domain (best val)")
    print(f"[done] log saved to {log_path}")


if __name__ == "__main__":
    main()
