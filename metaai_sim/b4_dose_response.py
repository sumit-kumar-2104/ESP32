"""
Phase 4b — B4 dose-response: how does cross-domain OTA accuracy degrade as
we inject increasing synthetic domain variation into the channel term?

The "domain" is represented as a per-recording complex multiplicative
factor applied at the channel output during EVALUATION (not training):
    y = |sum_i H_r(i) * x_i * c|      where c ~ CN(1, sigma^2 * I).

Concretely, we take the OTA_plain model trained on a single source room and
evaluate it on the SAME source room but with `c` drawn from an isotropic
complex Gaussian with variance `sigma^2` sweeping over --sigma-grid. The
result is a causal curve: cross-domain accuracy vs injected channel variance.

Rationale: real cross-room evaluation confounds many variables (users,
locations, RF environment). Injecting a controlled sigma isolates the
"channel drift" axis exactly, which is what the OTA path is supposed to
absorb via its complex weights but empirically does not.

Gated by Phase 1's assert_indomain_ok.

Output:
    results/b4_dose_response_<feature>_<dfs_bins>.png    — accuracy vs sigma
    results/b4_dose_response_<feature>_<dfs_bins>.csv    — sigma, mean, std

Usage:
    python b4_dose_response.py --feature dfs_spec --dfs-bins small \
        --dates 20181109 --seeds 42 123 7 --sigma-grid 0 0.05 0.1 0.2 0.4 0.8
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, print_device, set_seed, get_raw_csi_dir, require_raw_csi_dir
from data.csi_loader import (
    build_csi_features, FEATURE_MODES, DFS_BINS_MODES,
)
from models.linear_complex import ComplexLinear
from fix_pipeline import assert_indomain_ok

RESULTS_DIR = Path(__file__).parent / "results"

TRAIN_EPOCHS = 300
TRAIN_LR = 1e-3
WEIGHT_DECAY = 1e-4


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
        loss = ce(model(xc), yt)
        loss.backward()
        opt.step()
    return model


def eval_with_channel_noise(model, X, y, sigma, device, rng):
    """Evaluate acc with a per-sample multiplicative complex-Gaussian channel."""
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    xc = torch.complex(xt, torch.zeros_like(xt))
    with torch.no_grad():
        y_complex = torch.matmul(xc, model.complex_weight)   # (N, R)
        if sigma > 0:
            re = torch.tensor(rng.normal(1.0, sigma / (2 ** 0.5), size=y_complex.shape),
                              dtype=torch.float32, device=device)
            im = torch.tensor(rng.normal(0.0, sigma / (2 ** 0.5), size=y_complex.shape),
                              dtype=torch.float32, device=device)
            c = torch.complex(re, im)
            y_complex = y_complex * c
        preds = torch.abs(y_complex).argmax(dim=1).cpu().numpy()
    return accuracy_score(y, preds)


def load_dataset(args):
    csi_root = Path(args.csi_root) if args.csi_root else get_raw_csi_dir()
    if not csi_root.exists() or not any(csi_root.iterdir()):
        require_raw_csi_dir()
    data = build_csi_features(
        csi_root, args.dates,
        keep_users=set(args.users) if args.users else None,
        keep_gestures=set(args.gestures) if args.gestures else None,
        feature=args.feature, dfs_bins=args.dfs_bins,
    )
    return dict(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi-root", default=None)
    ap.add_argument("--feature", choices=FEATURE_MODES, default="dfs_spec")
    ap.add_argument("--dfs-bins", choices=DFS_BINS_MODES, default="small",
                    dest="dfs_bins")
    ap.add_argument("--dates", nargs="+", default=["20181109"],
                    help="single source-room date (in-domain training + eval); "
                         "channel noise supplies the domain shift.")
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    ap.add_argument("--sigma-grid", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5])
    ap.add_argument("--indomain-threshold", type=float, default=0.60)
    args = ap.parse_args()

    setup_logging("b4_dose_response")
    device = print_device()
    set_seed(args.seed)

    data = load_dataset(args)
    X = np.asarray(data["X"], dtype=np.float32)
    y_g_raw = np.asarray(data["y_gesture"])
    uniq_g = np.unique(y_g_raw)
    g_map = {g: i for i, g in enumerate(uniq_g)}
    y = np.array([g_map[g] for g in y_g_raw], dtype=np.int64)
    num_classes = len(uniq_g)
    print(f"[data] N={len(X)} dim={X.shape[1]} classes={num_classes}")

    accs = {sigma: [] for sigma in args.sigma_grid}
    gate_passed = False
    for seed in args.seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        sc = StandardScaler().fit(X_tr)
        Xtr_s = sc.transform(X_tr)
        Xte_s = sc.transform(X_te)
        model = train_ota_plain(Xtr_s, y_tr, X.shape[1], num_classes, device, seed)

        # Gate on the first seed (sigma=0).
        rng = np.random.default_rng(seed)
        clean = eval_with_channel_noise(model, Xte_s, y_te, 0.0, device, rng)
        if not gate_passed:
            print(f"[gate] OTA in-domain accuracy (seed={seed}) = {clean*100:.2f}%")
            try:
                assert_indomain_ok(clean, threshold=args.indomain_threshold,
                                   label="OTA in-domain accuracy for dose-response")
            except Exception as e:
                print(f"\n[GATE FAIL] {e}")
                sys.exit(2)
            gate_passed = True

        for sigma in args.sigma_grid:
            a = eval_with_channel_noise(model, Xte_s, y_te, sigma, device, rng)
            accs[sigma].append(a)
            print(f"  seed={seed}  sigma={sigma:.3f}  acc={a*100:.2f}%")

    # Aggregate
    means = np.array([np.mean(accs[s]) for s in args.sigma_grid])
    stds = np.array([np.std(accs[s]) for s in args.sigma_grid])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.feature}_{args.dfs_bins}"
    csv_path = RESULTS_DIR / f"b4_dose_response_{tag}.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("sigma,mean_acc,std_acc,n_seeds\n")
        for s, m, sd in zip(args.sigma_grid, means, stds):
            fh.write(f"{s:.4f},{m:.6f},{sd:.6f},{len(args.seeds)}\n")
    print(f"\n[saved] {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(args.sigma_grid, means * 100, yerr=stds * 100, marker="o",
                capsize=4, color="#4C72B0")
    chance = 100.0 / num_classes
    ax.axhline(chance, color="gray", linestyle="--",
               label=f"chance ({chance:.1f}%)")
    ax.set_xlabel("Injected channel-drift sigma (per-sample CN(1, sigma^2))")
    ax.set_ylabel("OTA test accuracy (%)")
    ax.set_title(f"B4 dose-response — accuracy vs channel drift [{tag}]")
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"b4_dose_response_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")

    print("\n[done] B4 dose-response complete.")


if __name__ == "__main__":
    main()
