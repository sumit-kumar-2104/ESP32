"""
Phase 4a — B3 CKA between in-domain and cross-domain OTA representations.

Computes linear Centered Kernel Alignment (CKA) between features extracted
from the in-domain training room and features extracted from the held-out
cross-domain room, at every stage of the OTA pipeline:

    stage A — raw input                       (StandardScaler-normalized)
    stage B — post-channel                    Re/Im of (x @ H) concatenated
    stage C — post-magnitude                  |x @ H|
    stage D — pre-decision                    same as C (single-layer OTA)

CKA is a similarity measure invariant to invertible linear transforms of
its two feature sets, so it fairly compares different geometries at each
stage. A drop from ~1.0 to near 0 marks the layer where the two rooms'
representations disperse — the stage where the "domain leak" becomes a
"domain shift". This localizes the argument that Proposition 1 addresses:
if similarity collapses at the magnitude readout (stage C/D), the decision
rule is the lever.

Output:
    results/b3_cka_<feature>_<dfs_bins>.png    — bar plot of stage-wise CKA
    results/b3_cka_<feature>_<dfs_bins>.csv    — machine-readable numbers

Gated by Phase 1's assert_indomain_ok (uses OTA_plain in-domain accuracy).

Usage:
    python b3_cka.py --feature dfs_spec --dfs-bins small --balance-room \
        --dates 20181109 20181118 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, print_device, set_seed, get_raw_csi_dir, require_raw_csi_dir
from data.csi_loader import (
    build_csi_features, balance_by_room, FEATURE_MODES, DFS_BINS_MODES,
)
from models.linear_complex import ComplexLinear
from fix_pipeline import assert_indomain_ok

RESULTS_DIR = Path(__file__).parent / "results"

TRAIN_EPOCHS = 300
TRAIN_LR = 1e-3
WEIGHT_DECAY = 1e-4


# ─── Linear CKA (Kornblith et al., 2019) ──────────────────────────────────────

def linear_cka(X, Y):
    """Linear CKA between two feature matrices of shape (n, d1), (n, d2).

    HSIC(K, L) numerator, using centered Gram matrices. Rows of X and Y
    correspond to the SAME samples so the two feature sets are aligned.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Frobenius norm formulation: |X^T Y|_F^2 / (|X^T X|_F * |Y^T Y|_F)
    num = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    if den < 1e-12:
        return float("nan")
    return float(num / den)


# ─── OTA train + stage extraction ─────────────────────────────────────────────

def train_ota_plain(X_tr, y_tr, in_dim, num_classes, device, seed):
    torch.manual_seed(seed)
    model = ComplexLinear(in_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR,
                           weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    xc = torch.complex(xt, torch.zeros_like(xt))
    model.train()
    for _ in range(TRAIN_EPOCHS):
        opt.zero_grad()
        logits = model(xc)
        loss = ce(logits, yt)
        loss.backward()
        opt.step()
    return model


def extract_stages(model, X, device):
    """Return dict of stage_name -> feature matrix on samples X."""
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    xc = torch.complex(xt, torch.zeros_like(xt))
    with torch.no_grad():
        y_complex = torch.matmul(xc, model.complex_weight)     # stage B
        y_mag = torch.abs(y_complex)                            # stage C
    stages = {
        "A raw input": X.astype(np.float32),
        "B post-channel (Re,Im)": np.concatenate(
            [y_complex.real.cpu().numpy(), y_complex.imag.cpu().numpy()], axis=1
        ),
        "C post-magnitude": y_mag.cpu().numpy(),
        "D pre-decision": y_mag.cpu().numpy(),
    }
    return stages


# ─── Data ─────────────────────────────────────────────────────────────────────

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi-root", default=None)
    ap.add_argument("--feature", choices=FEATURE_MODES, default="dfs_spec")
    ap.add_argument("--dfs-bins", choices=DFS_BINS_MODES, default="small",
                    dest="dfs_bins")
    ap.add_argument("--balance-room", action="store_true")
    ap.add_argument("--dates", nargs="+", default=["20181109", "20181118"])
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--indomain-threshold", "--target-acc", type=float,
                    default=0.50, dest="indomain_threshold",
                    help="honest DFS default 0.50; raw CSI: pass 0.70+.")
    args = ap.parse_args()

    setup_logging("b3_cka")
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
    if len(rooms) < 2:
        print("[FATAL] need >=2 rooms for CKA between in-dom and cross-dom.")
        sys.exit(1)
    src_room, tgt_room = rooms[0], rooms[1]
    print(f"[data] N={len(X)} dim={X.shape[1]}  source_room={src_room} "
          f"target_room={tgt_room}   per-room {dict(zip(rooms.tolist(), counts.tolist()))}")

    # Train OTA on source room only; keep two aligned batches for CKA.
    src_mask = y_room == src_room
    tgt_mask = y_room == tgt_room

    sc = StandardScaler().fit(X[src_mask])
    X_src_s = sc.transform(X[src_mask])
    X_tgt_s = sc.transform(X[tgt_mask])

    # Split source into train/test for the in-domain gate.
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_src_s, y[src_mask], test_size=0.2, random_state=args.seed,
        stratify=y[src_mask],
    )
    model = train_ota_plain(X_tr, y_tr, X.shape[1], num_classes, device, args.seed)
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_te, dtype=torch.float32, device=device)
        preds = model(torch.complex(xt, torch.zeros_like(xt))).argmax(dim=1)
    in_acc = float((preds.cpu().numpy() == y_te).mean())
    print(f"[gate] OTA in-domain accuracy = {in_acc*100:.2f}%")
    try:
        assert_indomain_ok(in_acc, threshold=args.indomain_threshold,
                           label="OTA in-domain accuracy for CKA")
    except Exception as e:
        print(f"\n[GATE FAIL] {e}")
        sys.exit(2)

    # Balance sample counts so CKA is computed on equal N.
    n_src = int(src_mask.sum())
    n_tgt = int(tgt_mask.sum())
    n = min(n_src, n_tgt)
    rng = np.random.default_rng(args.seed)
    src_idx = rng.choice(np.where(src_mask)[0], size=n, replace=False)
    tgt_idx = rng.choice(np.where(tgt_mask)[0], size=n, replace=False)
    X_src = sc.transform(X[src_idx])
    X_tgt = sc.transform(X[tgt_idx])

    stages_src = extract_stages(model, X_src, device)
    stages_tgt = extract_stages(model, X_tgt, device)

    stage_names = list(stages_src.keys())
    ckas = []
    print("\n" + "=" * 60)
    print("  Stage-wise linear CKA (in-domain vs cross-domain)")
    print("=" * 60)
    print(f"  {'stage':<28}{'CKA':>10}")
    for name in stage_names:
        c = linear_cka(stages_src[name], stages_tgt[name])
        ckas.append(c)
        print(f"  {name:<28}{c:>10.3f}")

    # Save plot + csv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.feature}_{args.dfs_bins}"
    csv_path = RESULTS_DIR / f"b3_cka_{tag}.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("stage,cka\n")
        for name, c in zip(stage_names, ckas):
            fh.write(f"{name},{c:.6f}\n")
    print(f"\n[saved] {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(stage_names))
    ax.bar(xs, ckas, color="#4C72B0", edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.split(" ", 1)[0] for n in stage_names])
    ax.set_ylabel("Linear CKA (source room vs target room)")
    ax.set_title(f"B3 CKA — where does similarity collapse? [{tag}]")
    for i, c in enumerate(ckas):
        ax.text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / f"b3_cka_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")

    print("\n[done] B3 CKA complete.")


if __name__ == "__main__":
    main()
