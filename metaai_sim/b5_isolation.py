"""
B5 — Isolation experiment: does the COMPUTATION (not the input) drive the
environment-specific collapse?

On the SAME balanced cross-room CSI dump and the SAME grouped cross-room split,
we vary ONLY the computation used to turn features into a gesture decision:

    (a) OTA_linear    — a complex linear layer followed by a magnitude |.|
                        readout, argmax over class scores. This mirrors the
                        MetaAI over-the-air forward pass (paper Eqn. 3:
                        y_r = |Σ H·x|).
    (b) Digital_LinMag — a NOISE-FREE OTA twin: one real-valued linear layer
                        W x (no bias), then a magnitude |.| readout, argmax
                        over class scores. Identical readout structure to
                        OTA_linear but trained with standard real-valued
                        backprop and NO channel noise / NO OTA constraints.
                        Isolates whether OTA_linear's failure to fit in-
                        domain is driven by the linear+magnitude paradigm
                        (expressivity) or by the wireless channel.
    (c) Digital_MLP   — a small real-valued MLP (Linear-ReLU-Linear-ReLU-
                        Linear) on the exact same standardized input.
    (d) Digital_DANN  — same MLP backbone as Digital_MLP, plus a room-domain
                        classifier head fed through a gradient-reversal layer
                        (Ganin & Lempitsky 2015). Trained jointly with the
                        gesture CE loss on the source room(s) and a domain-
                        invariance loss over source + (unlabeled) target
                        room, weighted by --lambda_dann (default 0.3) with a
                        linear warmup ramp over the first N epochs
                        (--lambda_dann_warmup_epochs, default 50).

Both models are trained to classify GESTURE. We report, mean ± std over
>= 3 seeds, for each model:
    - in-domain accuracy + macro-F1     (grouped 5-fold CV over pooled rooms)
    - cross-room accuracy + macro-F1    (train on room A, test on room B; both
                                         directions averaged)

Then we extract each model's pre-argmax / penultimate features on the held-out
folds and run the existing domain probe (room / location / user) on those
COMPUTED features (not the raw input). If the two computations retain different
amounts of room information from the SAME input, the computation itself is the
lever.

Outputs (results/):
    b5_isolation_summary.csv     — model x {in-dom acc, cross-room acc,
                                   room-decodability-from-computed-features}
                                   Rows are APPENDED with a `feature_mode`
                                   column so runs on different features don't
                                   overwrite each other.
    b5_confusion_<model>_<feature>.png / .npy — cross-room gesture confusion

HARD RULES honoured:
    - accepts --seed, prints the compute device, writes a timestamped log
    - never fabricates data: if the raw CSI is missing it fails loudly
    - does not alter existing default behaviour or the BVP paths

Usage:
    python b5_isolation.py --features csi --balance-room --feature dfs_spec \
        --dfs-bins small \
        --models OTA_linear Digital_LinMag Digital_MLP Digital_DANN \
        --lambda_dann 0.3 --seed 42
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
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, print_device, set_seed
from data.csi_loader import (
    build_csi_features,
    balance_by_room,
    FEATURE_MODES,
    DFS_BINS_MODES,
)
from models.linear_complex import ComplexLinear
# Reuse the EXISTING domain probe so the computed features are evaluated with
# the same methodology as b2.
from b2_probe import run_probe, logreg_factory

DUMPS_DIR = Path(__file__).parent / "dumps"
RESULTS_DIR = Path(__file__).parent / "results"

N_FOLDS = 5
TRAIN_EPOCHS = 300
TRAIN_LR = 1e-3
WEIGHT_DECAY = 1e-4
MLP_H1, MLP_H2 = 64, 32


# ─── Models (only the computation differs) ────────────────────────────────────

class DigitalMLP(nn.Module):
    """Small real-valued MLP. Penultimate = MLP_H2-dim pre-logit activations."""

    def __init__(self, in_dim, num_classes, h1=MLP_H1, h2=MLP_H2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, num_classes)

    def penultimate(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x):
        return self.out(self.penultimate(x))


class DigitalLinMag(nn.Module):
    """Noise-free OTA twin: one real-valued linear layer (no bias) then |.|.

    Computation:  scores_r = | (W x)_r | ; argmax_r scores_r.

    Same magnitude readout as OTA_linear, but with a real weight matrix and
    NO channel noise / NO OTA constraints — trained by standard backprop.
    This isolates whether OTA_linear's failure to fit in-domain is driven by
    the linear+magnitude paradigm itself (expressivity) or by the wireless
    channel model.
    """

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=False)

    def forward(self, x):
        return self.linear(x).abs()


class _GradReverse(torch.autograd.Function):
    """Gradient-reversal layer (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class DigitalDANN(nn.Module):
    """Digital_MLP backbone + room-domain head fed through a GRL.

    Same shared trunk as Digital_MLP so the computed features live in an
    identically-sized penultimate space; only the training objective differs.
    """

    def __init__(self, in_dim, num_classes, num_domains,
                 h1=MLP_H1, h2=MLP_H2, lambd=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, num_classes)
        self.domain_head = nn.Linear(h2, num_domains)
        self.lambd = float(lambd)

    def penultimate(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x):
        return self.out(self.penultimate(x))

    def forward_with_domain(self, x):
        z = self.penultimate(x)
        y = self.out(z)
        d = self.domain_head(grad_reverse(z, self.lambd))
        return y, d


# ─── Training / inference helpers ─────────────────────────────────────────────

def _train_model(kind, in_dim, num_classes, X_tr, y_tr, device, seed,
                 d_tr=None, X_unlab=None, d_unlab=None,
                 num_domains=None, lambda_dann=0.5,
                 lambda_dann_warmup_epochs=0):
    """Train one model on standardized real features and return it.

    For Digital_DANN, `d_tr` provides the source-domain labels (int), and
    (`X_unlab`, `d_unlab`) optionally provides an unlabeled target-domain batch
    used only by the domain head. Task loss is computed on X_tr only. If
    `lambda_dann_warmup_epochs > 0`, the GRL scaling is linearly ramped from
    0 to `lambda_dann` over the first N epochs (standard DANN practice) to
    let the gesture head learn a useful representation before the domain-
    invariance pressure kicks in.
    """
    torch.manual_seed(seed)
    if kind == "OTA_linear":
        model = ComplexLinear(in_dim, num_classes).to(device)
    elif kind == "Digital_LinMag":
        model = DigitalLinMag(in_dim, num_classes).to(device)
    elif kind == "Digital_MLP":
        model = DigitalMLP(in_dim, num_classes).to(device)
    elif kind == "Digital_DANN":
        if num_domains is None or num_domains < 2 or d_tr is None:
            # Fall back to plain MLP training if there aren't at least 2
            # observable domains — DANN needs contrast to be meaningful.
            model = DigitalDANN(in_dim, num_classes,
                                num_domains=max(int(num_domains or 1), 2),
                                lambd=lambda_dann).to(device)
        else:
            model = DigitalDANN(in_dim, num_classes, num_domains=int(num_domains),
                                lambd=lambda_dann).to(device)
    else:
        raise ValueError(kind)

    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)

    if kind == "OTA_linear":
        # OTA forward expects a complex symbol vector; the CSI feature is the
        # real part, imaginary part = 0 (mirrors modulating x onto the channel).
        xt_in = torch.complex(xt, torch.zeros_like(xt))
    else:
        xt_in = xt

    dann_active = (kind == "Digital_DANN" and d_tr is not None
                   and num_domains is not None and num_domains >= 2)
    if dann_active:
        dt = torch.tensor(d_tr, dtype=torch.long, device=device)
        if X_unlab is not None and d_unlab is not None and len(X_unlab) > 0:
            xu = torch.tensor(X_unlab, dtype=torch.float32, device=device)
            du = torch.tensor(d_unlab, dtype=torch.long, device=device)
            x_dom = torch.cat([xt_in, xu], dim=0)
            d_dom = torch.cat([dt, du], dim=0)
        else:
            x_dom = xt_in
            d_dom = dt

    warmup = max(0, int(lambda_dann_warmup_epochs))
    model.train()
    for epoch in range(TRAIN_EPOCHS):
        # Linear warmup of the GRL scaling from 0 to lambda_dann over the
        # first `warmup` epochs; constant at lambda_dann afterwards.
        if dann_active:
            if warmup > 0 and epoch < warmup:
                model.lambd = float(lambda_dann) * (epoch + 1) / warmup
            else:
                model.lambd = float(lambda_dann)
        opt.zero_grad()
        if dann_active:
            logits = model(xt_in)
            task_loss = loss_fn(logits, yt)
            _, dom_logits = model.forward_with_domain(x_dom)
            dom_loss = loss_fn(dom_logits, d_dom)
            loss = task_loss + dom_loss   # GRL already scales by lambd
        else:
            logits = model(xt_in)
            loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    return model


def _infer(kind, model, X, device):
    """Return (predictions, penultimate/pre-argmax features) as numpy arrays."""
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    if kind == "OTA_linear":
        xt = torch.complex(xt, torch.zeros_like(xt))
    with torch.no_grad():
        if kind == "OTA_linear" or kind == "Digital_LinMag":
            feats = model(xt)                     # pre-argmax |.| magnitudes
            logits = feats
        else:
            feats = model.penultimate(xt)         # penultimate activations
            logits = model.out(feats)
        preds = logits.argmax(dim=1)
    return preds.cpu().numpy(), feats.cpu().numpy()


def _grouped_splits(X, y, groups, seed, n_folds=N_FOLDS):
    if groups is not None and len(np.unique(groups)) >= n_folds:
        try:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return list(cv.split(X, y, groups))
        except ValueError:
            pass
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


# ─── Data loading (build-or-load, never fabricate) ────────────────────────────

def load_dataset(args, seed):
    """Return the balanced cross-room CSI dump as a dict of numpy arrays.

    Strategy: reuse dumps/<features>.npz if it was built with the SAME feature
    mode; otherwise rebuild it from the raw CSI so the requested --feature is
    honoured exactly. Never invents samples — a missing raw CSI tree fails loud.
    """
    npz_path = DUMPS_DIR / f"{args.features}.npz"
    data = None
    if npz_path.exists():
        loaded = np.load(npz_path, allow_pickle=True)
        mode = str(loaded["feature_mode"]) if "feature_mode" in loaded else "amp"
        dump_bins = str(loaded["dfs_bins"]) if "dfs_bins" in loaded else "full"
        wanted_bins = args.dfs_bins if args.feature == "dfs_spec" else "full"
        if mode == args.feature and dump_bins == wanted_bins:
            print(f"[data] reusing {npz_path} (feature_mode={mode}, dfs_bins={dump_bins})")
            data = {k: loaded[k] for k in loaded.files}
        else:
            print(f"[data] {npz_path} built with feature_mode={mode!r} "
                  f"dfs_bins={dump_bins!r} but --feature={args.feature!r} "
                  f"--dfs-bins={wanted_bins!r} requested — rebuilding from raw CSI.")

    if data is None:
        from config import get_data_dir
        csi_root = Path(args.csi_root) if args.csi_root else \
            get_data_dir() / "widar3" / "CSI"
        if not csi_root.exists():
            print("\n[FATAL] Raw CSI not found and no matching dump present.")
            print(f"        Expected CSI root at: {csi_root}")
            print("        Build it first, e.g.:")
            print(f"          python b2_dump_csi.py --feature {args.feature} "
                  f"--dfs-bins {args.dfs_bins} --balance-room --seed {args.seed}")
            print("        or pass --csi-root /path/to/widar3/CSI")
            sys.exit(1)
        print(f"[data] building CSI features from {csi_root} "
              f"(feature={args.feature}, dfs_bins={args.dfs_bins})")
        data = build_csi_features(
            csi_root, args.dates,
            keep_users=set(args.users) if args.users else None,
            keep_gestures=set(args.gestures) if args.gestures else None,
            feature=args.feature,
            dfs_bins=args.dfs_bins,
        )
        data = dict(data)

    if args.balance_room:
        data = balance_by_room(data, seed=seed)

    return data


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(kind, X, y_g, groups, y_room, seed, device,
                   lambda_dann=0.5, lambda_dann_warmup_epochs=0):
    """Run in-domain CV and cross-room evaluation for one model/seed.

    Returns a dict with in-domain acc/f1, cross-room acc/f1, aggregated
    cross-room confusion, and computed (penultimate) features for every sample
    (assembled from the held-out folds of the in-domain CV).
    """
    n, in_dim = X.shape
    num_classes = int(y_g.max()) + 1
    num_domains = int(np.unique(y_room).size)

    # ── in-domain: grouped 5-fold over the pooled (both-room) data ──
    comp_feats = None
    in_acc, in_f1 = [], []
    for tr, te in _grouped_splits(X, y_g, groups, seed):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # For DANN in-domain: training fold already spans both rooms, so the
        # invariance signal comes from the source's own domain labels. We do
        # NOT peek at test features (that would be an unfair leak vs the plain
        # MLP baseline).
        d_tr = y_room[tr] if kind == "Digital_DANN" else None
        model = _train_model(
            kind, in_dim, num_classes, Xtr, y_g[tr], device, seed,
            d_tr=d_tr, X_unlab=None, d_unlab=None,
            num_domains=num_domains, lambda_dann=lambda_dann,
            lambda_dann_warmup_epochs=lambda_dann_warmup_epochs,
        )
        preds, feats = _infer(kind, model, Xte, device)
        in_acc.append(accuracy_score(y_g[te], preds))
        in_f1.append(f1_score(y_g[te], preds, average="macro", zero_division=0))
        if comp_feats is None:
            comp_feats = np.zeros((n, feats.shape[1]), dtype=np.float32)
        comp_feats[te] = feats

    # ── cross-room: train on one room, test on the other (both directions) ──
    rooms = np.unique(y_room)
    cross_acc, cross_f1 = [], []
    cross_true, cross_pred = [], []
    if len(rooms) >= 2:
        for r_tr in rooms:
            tr = y_room == r_tr
            te = ~tr
            if tr.sum() == 0 or te.sum() == 0:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            # For DANN cross-room: source = training room (single label), target
            # = test room (unlabeled). Feed unlabeled test features to the
            # domain head; task loss is still gesture CE on the source only.
            d_tr = y_room[tr] if kind == "Digital_DANN" else None
            X_unlab = Xte if kind == "Digital_DANN" else None
            d_unlab = y_room[te] if kind == "Digital_DANN" else None
            model = _train_model(
                kind, in_dim, num_classes, Xtr, y_g[tr], device, seed,
                d_tr=d_tr, X_unlab=X_unlab, d_unlab=d_unlab,
                num_domains=num_domains, lambda_dann=lambda_dann,
                lambda_dann_warmup_epochs=lambda_dann_warmup_epochs,
            )
            preds, _ = _infer(kind, model, Xte, device)
            cross_acc.append(accuracy_score(y_g[te], preds))
            cross_f1.append(f1_score(y_g[te], preds, average="macro", zero_division=0))
            cross_true.append(y_g[te])
            cross_pred.append(preds)

    return {
        "in_acc": float(np.mean(in_acc)),
        "in_f1": float(np.mean(in_f1)),
        "cross_acc": float(np.mean(cross_acc)) if cross_acc else float("nan"),
        "cross_f1": float(np.mean(cross_f1)) if cross_f1 else float("nan"),
        "comp_feats": comp_feats,
        "cross_true": np.concatenate(cross_true) if cross_true else np.array([]),
        "cross_pred": np.concatenate(cross_pred) if cross_pred else np.array([]),
    }


def _save_confusion(kind, feature_mode, y_true, y_pred, num_classes):
    if y_true.size == 0:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tag = f"{kind}_{feature_mode}"
    np.save(RESULTS_DIR / f"b5_confusion_{tag}.npy", cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"B5 cross-room gesture confusion — {kind} [{feature_mode}]")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = RESULTS_DIR / f"b5_confusion_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  [saved] {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

ALL_MODELS = ["OTA_linear", "Digital_LinMag", "Digital_MLP", "Digital_DANN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="csi",
                    help="dump base name to reuse/build (default csi)")
    ap.add_argument("--feature", choices=FEATURE_MODES, default="amp",
                    help="CSI feature mode (default amp = original behaviour)")
    ap.add_argument("--dfs-bins", choices=DFS_BINS_MODES, default="full",
                    dest="dfs_bins",
                    help="dfs_spec size regime: full (default, 1536-dim) or "
                         "small (~150-dim compact low-Doppler band). Only "
                         "affects --feature dfs_spec; ignored otherwise.")
    ap.add_argument("--balance-room", action="store_true",
                    help="balance rooms so chance ~50%% before training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csi-root", default=None,
                    help="raw Widar3.0 CSI root (used only if a rebuild is needed)")
    ap.add_argument("--dates", nargs="+", default=["20181109", "20181118"])
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS,
                    default=["OTA_linear", "Digital_MLP"],
                    help="which models to evaluate (any subset of "
                         "OTA_linear, Digital_LinMag, Digital_MLP, "
                         "Digital_DANN). Default keeps the original two so "
                         "existing runs are unchanged.")
    ap.add_argument("--lambda_dann", type=float, default=0.3,
                    help="GRL scaling target for Digital_DANN's domain-"
                         "invariance loss (default 0.3, sweep by changing "
                         "this flag).")
    ap.add_argument("--lambda_dann_warmup_epochs", type=int, default=50,
                    help="Number of epochs over which to linearly ramp the "
                         "GRL scaling from 0 to --lambda_dann (default 50; "
                         "pass 0 to disable warmup).")
    args = ap.parse_args()

    setup_logging("b5_isolation")
    device = print_device()
    set_seed(args.seed)

    seeds = [args.seed, args.seed + 1, args.seed + 2]  # >= 3 seeds
    print("=" * 70)
    print("B5 — Isolation experiment (computation is the only varying factor)")
    print(f"  features={args.features} feature={args.feature} "
          f"dfs_bins={args.dfs_bins} balance_room={args.balance_room} seeds={seeds}")
    print(f"  models={args.models} lambda_dann={args.lambda_dann} "
          f"warmup_epochs={args.lambda_dann_warmup_epochs}")
    print("=" * 70)

    data = load_dataset(args, seed=args.seed)
    X = np.asarray(data["X"], dtype=np.float32)
    groups = np.asarray(data["groups"])
    y_room = np.asarray(data["y_room"])
    y_location = np.asarray(data["y_location"])
    y_user = np.asarray(data["y_user"])

    # Remap gesture labels to a contiguous 0..C-1 range for CrossEntropy.
    y_g_raw = np.asarray(data["y_gesture"])
    uniq_g = np.unique(y_g_raw)
    g_map = {g: i for i, g in enumerate(uniq_g)}
    y_g = np.array([g_map[g] for g in y_g_raw], dtype=np.int64)
    num_classes = len(uniq_g)

    rooms, room_counts = np.unique(y_room, return_counts=True)
    chance_room = room_counts.max() / room_counts.sum()
    print(f"[data] N={len(X)} dim={X.shape[1]} gestures={num_classes} "
          f"rooms={dict(zip(rooms.tolist(), room_counts.tolist()))}")
    print(f"[data] room chance level = {chance_room * 100:.1f}%")
    if len(rooms) < 2:
        print("[FATAL] need >= 2 rooms for a cross-room experiment.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve user-requested order but drop dupes.
    seen = set()
    models = [m for m in args.models if not (m in seen or seen.add(m))]
    agg = {m: {"in_acc": [], "in_f1": [], "cross_acc": [], "cross_f1": []}
           for m in models}
    # collect cross-room predictions across seeds for the confusion matrices
    conf = {m: {"true": [], "pred": []} for m in models}
    # room-decodability from computed features, per seed
    room_dec = {m: [] for m in models}
    loc_dec = {m: [] for m in models}
    user_dec = {m: [] for m in models}

    for seed in seeds:
        print(f"\n{'─' * 60}\n  SEED {seed}\n{'─' * 60}")
        for m in models:
            res = evaluate_model(m, X, y_g, groups, y_room, seed, device,
                                 lambda_dann=args.lambda_dann,
                                 lambda_dann_warmup_epochs=args.lambda_dann_warmup_epochs)
            agg[m]["in_acc"].append(res["in_acc"])
            agg[m]["in_f1"].append(res["in_f1"])
            agg[m]["cross_acc"].append(res["cross_acc"])
            agg[m]["cross_f1"].append(res["cross_f1"])
            conf[m]["true"].append(res["cross_true"])
            conf[m]["pred"].append(res["cross_pred"])
            print(f"  [{m}] in-dom acc={res['in_acc']*100:.1f}% "
                  f"F1={res['in_f1']:.3f} | cross-room acc={res['cross_acc']*100:.1f}% "
                  f"F1={res['cross_f1']:.3f}")

            # Domain probe on the COMPUTED features (room/location/user).
            cf = res["comp_feats"]
            ra, _, _, _ = run_probe(cf, y_room, groups, logreg_factory)
            la, _, _, _ = run_probe(cf, y_location, groups, logreg_factory)
            ua, _, _, _ = run_probe(cf, y_user, groups, logreg_factory)
            room_dec[m].append(ra)
            loc_dec[m].append(la)
            user_dec[m].append(ua)
            print(f"       computed-feature decodability: "
                  f"room={ra*100:.1f}% loc={la*100:.1f}% user={ua*100:.1f}%")

    # ── confusion matrices (aggregated over seeds/directions) ──
    print(f"\n{'=' * 70}\n  Saving confusion matrices\n{'=' * 70}")
    for m in models:
        yt = np.concatenate([a for a in conf[m]["true"] if a.size]) \
            if any(a.size for a in conf[m]["true"]) else np.array([])
        yp = np.concatenate([a for a in conf[m]["pred"] if a.size]) \
            if any(a.size for a in conf[m]["pred"]) else np.array([])
        _save_confusion(m, args.feature, yt, yp, num_classes)

    # ── summary table ──
    def ms(v):
        v = np.asarray(v, dtype=float)
        return f"{np.nanmean(v)*100:.1f}±{np.nanstd(v)*100:.1f}"

    print(f"\n{'=' * 70}")
    print(f"  B5 ISOLATION SUMMARY  (mean±std over seeds {seeds})")
    dfs_note = f" dfs_bins = {args.dfs_bins}" if args.feature == "dfs_spec" else ""
    print(f"  feature_mode = {args.feature}{dfs_note}  |  "
          f"room chance level = {chance_room*100:.1f}%")
    print(f"{'=' * 70}")
    header = (f"  {'model':<15} {'in-dom acc':<14} {'in-dom F1':<14} "
              f"{'cross acc':<14} {'cross F1':<14} {'gap(in-cr)':<14} "
              f"{'room-decod':<12}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    header_cols = ("feature_mode,dfs_bins,model,in_dom_acc,in_dom_f1,"
                   "cross_room_acc,cross_room_f1,in_dom_minus_cross,"
                   "room_decodability,loc_decodability,"
                   "user_decodability,room_chance,lambda_dann")
    rows_out = []
    dfs_bins_col = args.dfs_bins if args.feature == "dfs_spec" else "n/a"
    for m in models:
        in_acc, in_f1 = ms(agg[m]["in_acc"]), ms(agg[m]["in_f1"])
        cr_acc, cr_f1 = ms(agg[m]["cross_acc"]), ms(agg[m]["cross_f1"])
        # Per-seed generalization gap so the mean±std correctly reflects
        # sample-level uncertainty (not the difference of two independent
        # aggregates).
        gap_vals = np.asarray(agg[m]["in_acc"], dtype=float) - \
            np.asarray(agg[m]["cross_acc"], dtype=float)
        gap = ms(gap_vals)
        rd = ms(room_dec[m])
        ld, ud = ms(loc_dec[m]), ms(user_dec[m])
        print(f"  {m:<15} {in_acc:<14} {in_f1:<14} {cr_acc:<14} {cr_f1:<14} "
              f"{gap:<14} {rd:<12}")
        lam = f"{args.lambda_dann}" if m == "Digital_DANN" else ""
        rows_out.append(
            f"{args.feature},{dfs_bins_col},{m},{in_acc},{in_f1},{cr_acc},{cr_f1},"
            f"{gap},{rd},{ld},{ud},{chance_room*100:.1f},{lam}"
        )

    # Append rows (with a `feature_mode` column) instead of overwriting, so
    # runs on different features don't clobber each other.
    summary_path = RESULTS_DIR / "b5_isolation_summary.csv"
    file_exists = summary_path.exists() and summary_path.stat().st_size > 0
    with open(summary_path, "a", encoding="utf-8") as fh:
        if not file_exists:
            fh.write(header_cols + "\n")
        for r in rows_out:
            fh.write(r + "\n")
    print(f"\n  [{'appended' if file_exists else 'saved'}] {summary_path}")
    print("\n[done] B5 isolation complete.")


if __name__ == "__main__":
    main()
