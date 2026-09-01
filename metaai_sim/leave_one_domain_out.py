"""
Phase 5 — Leave-one-domain-out (LODO) sweep for cross-domain generalization.

Existing scripts (`b5_isolation.py`, `c1_gate.py`) treat "cross-domain" as
the two-room contrast between date `20181109` and `20181118`. That's a
single held-out room and it hides between-room variability.

This script:

    - accepts >=2 dates via `--dates` and treats each unique room as a
      distinct domain (per `data/csi_loader.py::date_to_room`);
    - runs a leave-one-domain-out (LODO) loop: for each room D_i, train on
      the union of the OTHER rooms and evaluate on D_i;
    - repeats for `--seeds >= 3` and reports mean±std of cross-domain
      accuracy per model and per held-out room, plus an overall mean;
    - supports the same model zoo as b5_isolation.py: OTA_linear,
      Digital_LinMag, Digital_MLP, Digital_DANN;
    - is gated behind `assert_indomain_ok` using the pooled-source in-domain
      accuracy of OTA_linear before printing any cross-room number.

Nothing about the existing 2-date scripts is changed. This is an additive
Phase-5 wrapper. It reuses the training helpers from `b5_isolation.py` so
that the SAME computation is exercised in both scripts.

Usage:
    python leave_one_domain_out.py --feature dfs_spec --dfs-bins small \
        --dates 20181109 20181117 20181118 --balance-room \
        --models OTA_linear Digital_MLP Digital_DANN --seeds 42 123 7

Outputs:
    results/lodo_summary.csv      — appended rows: (feature, model, held-out
                                    room, mean±std cross-domain acc / F1)
    results/lodo_bar_<feature>.png — bar plot of per-model overall LODO acc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, print_device, set_seed, get_raw_csi_dir, require_raw_csi_dir
from data.csi_loader import (
    build_csi_features, balance_by_room, FEATURE_MODES, DFS_BINS_MODES,
)
from fix_pipeline import assert_indomain_ok
from b5_isolation import _train_model, _infer, ALL_MODELS

RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(args, seed):
    csi_root = Path(args.csi_root) if args.csi_root else get_raw_csi_dir()
    if not csi_root.exists() or not any(csi_root.iterdir()):
        require_raw_csi_dir()
    data = build_csi_features(
        csi_root, args.dates,
        keep_users=set(args.users) if args.users else None,
        keep_gestures=set(args.gestures) if args.gestures else None,
        feature=args.feature, dfs_bins=args.dfs_bins,
    )
    data = dict(data)
    if args.balance_room:
        data = balance_by_room(data, seed=seed)
    return data


def gate_indomain(X, y, seed, device, num_classes, threshold):
    """Pool source rooms, do 80/20 split, train OTA_linear plain, gate."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    sc = StandardScaler().fit(X_tr)
    Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
    model = _train_model("OTA_linear", X.shape[1], num_classes, Xtr, y_tr,
                         device, seed)
    preds, _ = _infer("OTA_linear", model, Xte, device)
    acc = accuracy_score(y_te, preds)
    print(f"[gate] OTA_linear pooled in-domain acc = {acc*100:.2f}%")
    assert_indomain_ok(acc, threshold=threshold,
                       label="OTA_linear pooled in-domain accuracy")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi-root", default=None)
    ap.add_argument("--feature", choices=FEATURE_MODES, default="dfs_spec")
    ap.add_argument("--dfs-bins", choices=DFS_BINS_MODES, default="small",
                    dest="dfs_bins")
    ap.add_argument("--balance-room", action="store_true")
    ap.add_argument("--dates", nargs="+",
                    default=["20181109", "20181117", "20181118"])
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS,
                    default=["OTA_linear", "Digital_MLP", "Digital_DANN"])
    ap.add_argument("--lambda-dann", type=float, default=0.3,
                    dest="lambda_dann")
    ap.add_argument("--lambda-dann-warmup-epochs", type=int, default=50,
                    dest="lambda_dann_warmup_epochs")
    ap.add_argument("--indomain-threshold", type=float, default=0.60)
    args = ap.parse_args()

    setup_logging("leave_one_domain_out")
    device = print_device()
    set_seed(args.seed)

    data = load_dataset(args, seed=args.seed)
    X = np.asarray(data["X"], dtype=np.float32)
    y_room = np.asarray(data["y_room"])
    y_g_raw = np.asarray(data["y_gesture"])
    uniq_g = np.unique(y_g_raw)
    g_map = {g: i for i, g in enumerate(uniq_g)}
    y = np.array([g_map[g] for g in y_g_raw], dtype=np.int64)
    num_classes = len(uniq_g)

    rooms, counts = np.unique(y_room, return_counts=True)
    print(f"[data] N={len(X)} dim={X.shape[1]}  rooms={dict(zip(rooms.tolist(), counts.tolist()))}")
    if len(rooms) < 2:
        print("[FATAL] LODO needs >=2 rooms.")
        sys.exit(1)
    if len(rooms) == 2:
        print("[NOTE] Only 2 rooms present; LODO reduces to two directional splits.")

    # Gate
    gate_indomain(X, y, args.seeds[0], device, num_classes,
                  threshold=args.indomain_threshold)

    # LODO sweep
    per_model = {m: {"per_room_acc": {}, "per_room_f1": {}, "overall": []}
                 for m in args.models}

    for seed in args.seeds:
        print(f"\n{'─'*60}\n  SEED {seed}\n{'─'*60}")
        for r_te in rooms:
            te = y_room == r_te
            tr = ~te
            if tr.sum() == 0 or te.sum() == 0:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            for m in args.models:
                d_tr = y_room[tr] if m == "Digital_DANN" else None
                X_unlab = Xte if m == "Digital_DANN" else None
                d_unlab = y_room[te] if m == "Digital_DANN" else None
                num_domains = int(np.unique(y_room).size)
                model = _train_model(
                    m, X.shape[1], num_classes, Xtr, y[tr], device, seed,
                    d_tr=d_tr, X_unlab=X_unlab, d_unlab=d_unlab,
                    num_domains=num_domains, lambda_dann=args.lambda_dann,
                    lambda_dann_warmup_epochs=args.lambda_dann_warmup_epochs,
                )
                preds, _ = _infer(m, model, Xte, device)
                a = accuracy_score(y[te], preds)
                f = f1_score(y[te], preds, average="macro", zero_division=0)
                per_model[m]["per_room_acc"].setdefault(int(r_te), []).append(a)
                per_model[m]["per_room_f1"].setdefault(int(r_te), []).append(f)
                per_model[m]["overall"].append(a)
                print(f"  [{m:<15}] held-out room={r_te} acc={a*100:5.1f}% "
                      f"F1={f:.3f}")

    def ms(v):
        v = np.asarray(v, dtype=float)
        return f"{np.nanmean(v)*100:.1f}±{np.nanstd(v)*100:.1f}"

    # Print + save
    print(f"\n{'='*70}\n  LODO SUMMARY (mean±std over seeds {args.seeds})")
    print(f"  feature = {args.feature}  dfs_bins = {args.dfs_bins}  "
          f"rooms = {rooms.tolist()}")
    print(f"{'='*70}")
    header = f"  {'model':<15}"
    for r in rooms:
        header += f"held-out={r}    "
    header += "overall"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in args.models:
        row = f"  {m:<15}"
        for r in rooms:
            vals = per_model[m]["per_room_acc"].get(int(r), [])
            row += f"{ms(vals):<15}"
        row += ms(per_model[m]["overall"])
        print(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "lodo_summary.csv"
    header_needed = not (csv_path.exists() and csv_path.stat().st_size > 0)
    with open(csv_path, "a", encoding="utf-8") as fh:
        if header_needed:
            fh.write("feature,dfs_bins,model,held_out_room,cross_acc,cross_f1,seeds\n")
        for m in args.models:
            for r in rooms:
                acc = per_model[m]["per_room_acc"].get(int(r), [])
                f1s = per_model[m]["per_room_f1"].get(int(r), [])
                fh.write(f"{args.feature},{args.dfs_bins},{m},{int(r)},"
                         f"{ms(acc)},{ms(f1s)},"
                         f"{'/'.join(str(s) for s in args.seeds)}\n")
    print(f"\n[saved] {csv_path}")

    # Overall bar plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = [np.mean(per_model[m]["overall"]) * 100 for m in args.models]
    stds = [np.std(per_model[m]["overall"]) * 100 for m in args.models]
    xs = np.arange(len(args.models))
    ax.bar(xs, means, yerr=stds, capsize=5, color="#4C72B0", edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(args.models, rotation=0)
    ax.set_ylabel("LODO cross-domain accuracy (%)")
    ax.set_title(f"Phase 5 LODO — overall cross-domain acc "
                 f"[{args.feature}, {args.dfs_bins}, {len(rooms)} rooms]")
    ax.set_ylim(0, 100)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + 1.5, f"{m:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / f"lodo_bar_{args.feature}_{args.dfs_bins}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")

    print("\n[done] LODO complete.")


if __name__ == "__main__":
    main()
