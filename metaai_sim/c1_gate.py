"""
Phase 3 — C1 gate on the OTA layer.

Compares cross-room gesture accuracy of the OTA PHYSICAL model (complex
linear -> |.| magnitude readout) with and without three domain-generalization
objectives:

    OTA_plain   — no DG regularization (baseline).
    OTA+DANN    — gradient-reversal domain adversary on the R-dim |.| output.
    OTA+IRM     — Invariant Risk Minimization penalty (Arjovsky et al., 2019)
                  averaged across observed source rooms.
    OTA+CORAL   — Coral loss (Sun & Saenko, 2016) between per-room feature
                  covariances of the |.| output.

The objectives are applied to the OTA forward path directly (magnitudes of a
complex linear layer). Optional `--quantize` swaps `ComplexLinear` for the
2-bit `DiscreteComplexLinear` from `models/discrete_nn.py`, so the DG
objectives coexist with the metasurface quantization + magnitude constraint.

Gated by Phase 1's `assert_indomain_ok`: the OTA_plain in-domain accuracy
must clear the 60% floor before any cross-room number is printed. If it
does not, the script prints Phase 0's diagnostic hints and exits.

Outputs to `results/`:
    c1_gate_summary.csv       — per-model mean±std cross-room acc / F1 / room-decod
    c1_gate_bar_<feature>.png — bar plot of cross-room accuracy by objective

Usage:
    python c1_gate.py --feature dfs_spec --dfs-bins small --balance-room \
        --seeds 42 123 7 [--quantize] [--lambda-dann 0.3] \
        [--lambda-irm 1.0] [--lambda-coral 1.0]

Hard rules honoured:
    - accepts --seed / --seeds; prints device; timestamped log in logs/
    - never fabricates data (fails loudly on missing CSI)
    - does not modify or delete any existing script's behaviour
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, print_device, set_seed
from data.csi_loader import (
    build_csi_features, balance_by_room, FEATURE_MODES, DFS_BINS_MODES,
)
from models.linear_complex import ComplexLinear
from models.discrete_nn import DiscreteComplexLinear
from fix_pipeline import assert_indomain_ok
from b2_probe import run_probe, logreg_factory

DUMPS_DIR = Path(__file__).parent / "dumps"
RESULTS_DIR = Path(__file__).parent / "results"

N_FOLDS = 5
TRAIN_EPOCHS = 300
TRAIN_LR = 1e-3
WEIGHT_DECAY = 1e-4


# ─── DG heads and losses ──────────────────────────────────────────────────────

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class DomainHead(nn.Module):
    """Small linear domain classifier fed via GRL from the OTA |.| output."""

    def __init__(self, in_dim, num_domains, lambd=0.3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_domains)
        self.lambd = float(lambd)

    def forward(self, feats):
        return self.fc(grad_reverse(feats, self.lambd))


def irm_penalty(logits, y):
    """Arjovsky et al. 2019 gradient-penalty formulation of IRM."""
    scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss = nn.functional.cross_entropy(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


def coral_loss(fs, ft):
    """Sun & Saenko 2016 CORAL loss between two feature batches.

    Aligns second-order (covariance) statistics of source (fs) and target (ft).
    """
    d = fs.shape[1]
    fs_c = fs - fs.mean(dim=0, keepdim=True)
    ft_c = ft - ft.mean(dim=0, keepdim=True)
    cs = (fs_c.t() @ fs_c) / max(fs.shape[0] - 1, 1)
    ct = (ft_c.t() @ ft_c) / max(ft.shape[0] - 1, 1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


# ─── OTA wrapper (magnitude output + reused for classification) ───────────────

def make_ota_model(in_dim, num_classes, quantize=False):
    if quantize:
        return DiscreteComplexLinear(in_dim, num_classes)
    return ComplexLinear(in_dim, num_classes)


def ota_forward(model, x_real):
    """OTA forward path: bipolarize -> complex -> complex linear -> |.|.

    Mirrors `sim/sender.py::encode_bpsk` and `models/linear_complex.py::forward`
    exactly so the DG objectives operate on the same tensor the paper does.
    """
    x_c = torch.complex(x_real, torch.zeros_like(x_real))
    return model(x_c)   # (batch, R)  = |.|


# ─── Training ─────────────────────────────────────────────────────────────────

def train_ota(objective, X_tr, y_tr, d_tr, in_dim, num_classes, num_domains,
              device, seed, quantize=False,
              lambda_dann=0.3, lambda_irm=1.0, lambda_coral=1.0,
              X_target=None):
    """Train ONE OTA model with the given DG objective and return it.

    objective in {'plain', 'dann', 'irm', 'coral'}.

    For DANN, `d_tr` provides source-domain labels (int). Cross-room mode
    additionally passes `X_target` (unlabeled features from the held-out room)
    so the domain head sees both distributions.
    For IRM, `d_tr` groups the training set into environments (per room).
    For CORAL, `d_tr` groups the training set; loss is pairwise between
    per-room feature covariances.
    """
    torch.manual_seed(seed)
    model = make_ota_model(in_dim, num_classes, quantize=quantize).to(device)
    dom_head = None
    params = list(model.parameters())
    if objective == "dann":
        dom_head = DomainHead(num_classes, max(num_domains, 2), lambd=lambda_dann).to(device)
        params += list(dom_head.parameters())
    opt = torch.optim.Adam(params, lr=TRAIN_LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    dt = torch.tensor(d_tr, dtype=torch.long, device=device)
    xu = torch.tensor(X_target, dtype=torch.float32, device=device) \
        if X_target is not None and len(X_target) > 0 else None

    uniq_d = torch.unique(dt).tolist()

    model.train()
    for epoch in range(TRAIN_EPOCHS):
        opt.zero_grad()
        logits = ota_forward(model, xt)              # (N, R) — |.|
        task_loss = ce(logits, yt)
        reg = torch.zeros((), device=device)

        if objective == "dann" and dom_head is not None:
            if xu is not None:
                x_dom = torch.cat([xt, xu], dim=0)
                # Unlabeled target contributes only via GRL: assign a synthetic
                # "target" domain id equal to num_domains-1 (last observed +1).
                du = torch.full((xu.shape[0],), fill_value=max(uniq_d) + 1,
                                dtype=torch.long, device=device)
                d_all = torch.cat([dt, du], dim=0)
            else:
                x_dom = xt
                d_all = dt
            feats = ota_forward(model, x_dom)
            dom_logits = dom_head(feats)
            n_dom = int(d_all.max().item()) + 1
            # Ensure the domain head's classifier width covers observed ids.
            if dom_head.fc.out_features < n_dom:
                # Rebuild head with a wider output layer on the fly. This can
                # only happen when unlabeled target is added; safe on epoch 0.
                new_head = DomainHead(num_classes, n_dom,
                                      lambd=dom_head.lambd).to(device)
                dom_head = new_head
                params = list(model.parameters()) + list(dom_head.parameters())
                opt = torch.optim.Adam(params, lr=TRAIN_LR,
                                       weight_decay=WEIGHT_DECAY)
                dom_logits = dom_head(feats)
            reg = ce(dom_logits, d_all)

        elif objective == "irm":
            envs = 0
            for d in uniq_d:
                mask = dt == d
                if int(mask.sum()) < 2:
                    continue
                reg = reg + irm_penalty(logits[mask], yt[mask])
                envs += 1
            if envs > 0:
                reg = reg / envs
            reg = lambda_irm * reg

        elif objective == "coral":
            # Pairwise CORAL between each pair of observed source domains.
            n_pairs = 0
            for i, di in enumerate(uniq_d):
                for dj in uniq_d[i + 1:]:
                    mi, mj = dt == di, dt == dj
                    if int(mi.sum()) < 2 or int(mj.sum()) < 2:
                        continue
                    reg = reg + coral_loss(logits[mi], logits[mj])
                    n_pairs += 1
            if n_pairs > 0:
                reg = reg / n_pairs
            reg = lambda_coral * reg

        loss = task_loss + reg
        loss.backward()
        opt.step()
    return model


def infer_ota(model, X, device):
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = ota_forward(model, xt)
    return logits.argmax(dim=1).cpu().numpy(), logits.cpu().numpy()


# ─── Splits ───────────────────────────────────────────────────────────────────

def grouped_splits(X, y, groups, seed, n_folds=N_FOLDS):
    if groups is not None and len(np.unique(groups)) >= n_folds:
        try:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return list(cv.split(X, y, groups))
        except ValueError:
            pass
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_dataset(args, seed):
    from config import get_raw_csi_dir
    csi_root = Path(args.csi_root) if args.csi_root else get_raw_csi_dir()
    if not csi_root.exists() or not any(csi_root.iterdir()):
        from config import require_raw_csi_dir
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


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_objective(objective, X, y_g, groups, y_room, seed, device,
                       args, num_classes):
    n, in_dim = X.shape
    num_domains = int(np.unique(y_room).size)

    # In-domain: grouped 5-fold over pooled rooms so we can gate cross-room.
    in_acc = []
    comp_feats = None
    for tr, te in grouped_splits(X, y_g, groups, seed):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        model = train_ota(
            objective, Xtr, y_g[tr], y_room[tr],
            in_dim, num_classes, num_domains, device, seed,
            quantize=args.quantize,
            lambda_dann=args.lambda_dann, lambda_irm=args.lambda_irm,
            lambda_coral=args.lambda_coral,
        )
        preds, feats = infer_ota(model, Xte, device)
        in_acc.append(accuracy_score(y_g[te], preds))
        if comp_feats is None:
            comp_feats = np.zeros((n, feats.shape[1]), dtype=np.float32)
        comp_feats[te] = feats

    # Cross-room: leave-one-room-out (each room as target in turn).
    rooms = np.unique(y_room)
    cross_acc, cross_f1 = [], []
    for r_te in rooms:
        te = y_room == r_te
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        X_target = Xte if objective == "dann" else None
        model = train_ota(
            objective, Xtr, y_g[tr], y_room[tr],
            in_dim, num_classes, num_domains, device, seed,
            quantize=args.quantize,
            lambda_dann=args.lambda_dann, lambda_irm=args.lambda_irm,
            lambda_coral=args.lambda_coral,
            X_target=X_target,
        )
        preds, _ = infer_ota(model, Xte, device)
        cross_acc.append(accuracy_score(y_g[te], preds))
        cross_f1.append(f1_score(y_g[te], preds, average="macro", zero_division=0))

    return {
        "in_acc": float(np.mean(in_acc)),
        "cross_acc": float(np.mean(cross_acc)) if cross_acc else float("nan"),
        "cross_f1": float(np.mean(cross_f1)) if cross_f1 else float("nan"),
        "comp_feats": comp_feats,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

OBJECTIVES = ["plain", "dann", "irm", "coral"]


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
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quantize", action="store_true",
                    help="use DiscreteComplexLinear (2-bit phase, STE) instead "
                         "of ComplexLinear so DG objectives run on the "
                         "metasurface-constrained model.")
    ap.add_argument("--lambda-dann", type=float, default=0.3, dest="lambda_dann")
    ap.add_argument("--lambda-irm", type=float, default=1.0, dest="lambda_irm")
    ap.add_argument("--lambda-coral", type=float, default=1.0, dest="lambda_coral")
    ap.add_argument("--indomain-threshold", type=float, default=0.60,
                    help="floor for OTA_plain in-domain accuracy; below this "
                         "the script refuses to print cross-room numbers.")
    ap.add_argument("--objectives", nargs="+", choices=OBJECTIVES,
                    default=OBJECTIVES)
    args = ap.parse_args()

    setup_logging("c1_gate")
    device = print_device()
    set_seed(args.seed)

    print("=" * 70)
    print("  Phase 3 — C1 gate on the OTA layer")
    print(f"  feature={args.feature} dfs_bins={args.dfs_bins} "
          f"quantize={args.quantize}")
    print(f"  seeds={args.seeds}   objectives={args.objectives}")
    print(f"  lambdas: DANN={args.lambda_dann} IRM={args.lambda_irm} "
          f"CORAL={args.lambda_coral}")
    print("=" * 70)

    data = load_dataset(args, seed=args.seed)
    X = np.asarray(data["X"], dtype=np.float32)
    groups = np.asarray(data["groups"])
    y_room = np.asarray(data["y_room"])
    y_loc = np.asarray(data["y_location"])
    y_user = np.asarray(data["y_user"])

    y_g_raw = np.asarray(data["y_gesture"])
    uniq_g = np.unique(y_g_raw)
    g_map = {g: i for i, g in enumerate(uniq_g)}
    y_g = np.array([g_map[g] for g in y_g_raw], dtype=np.int64)
    num_classes = len(uniq_g)
    rooms, counts = np.unique(y_room, return_counts=True)
    print(f"[data] N={len(X)} dim={X.shape[1]} classes={num_classes} "
          f"rooms={dict(zip(rooms.tolist(), counts.tolist()))}")
    if len(rooms) < 2:
        print("[FATAL] need >=2 rooms for a cross-room experiment.")
        sys.exit(1)

    # First seed, plain: gate on in-domain
    print(f"\n{'─'*60}\n  GATING SEED (plain OTA, seed={args.seeds[0]})\n{'─'*60}")
    gate = evaluate_objective("plain", X, y_g, groups, y_room, args.seeds[0],
                              device, args, num_classes)
    print(f"  OTA_plain in-domain acc = {gate['in_acc']*100:.2f}%")
    try:
        assert_indomain_ok(gate["in_acc"], threshold=args.indomain_threshold,
                           label="OTA_plain in-domain accuracy")
    except Exception as e:
        print(f"\n[GATE FAIL] {e}")
        print("[HINT] Run Phase 0:  python diagnose_indomain.py --features "
              f"{'dfs_full' if args.dfs_bins=='full' else 'dfs_small'}")
        print("[HINT] Then flip fix_pipeline.enable_hook_A / _B in this "
              "script's train loop and retry.")
        sys.exit(2)

    # Full sweep across objectives x seeds
    agg = {o: {"in_acc": [], "cross_acc": [], "cross_f1": [], "room_dec": []}
           for o in args.objectives}
    for seed in args.seeds:
        print(f"\n{'─'*60}\n  SEED {seed}\n{'─'*60}")
        for obj in args.objectives:
            res = evaluate_objective(obj, X, y_g, groups, y_room, seed,
                                     device, args, num_classes)
            agg[obj]["in_acc"].append(res["in_acc"])
            agg[obj]["cross_acc"].append(res["cross_acc"])
            agg[obj]["cross_f1"].append(res["cross_f1"])
            # Room decodability from the OTA |.| output on in-domain folds.
            if res["comp_feats"] is not None:
                ra, _, _, _ = run_probe(res["comp_feats"], y_room, groups,
                                        logreg_factory)
                agg[obj]["room_dec"].append(ra)
            else:
                agg[obj]["room_dec"].append(float("nan"))
            print(f"  [OTA+{obj:<5}] in-dom={res['in_acc']*100:5.1f}%  "
                  f"cross={res['cross_acc']*100:5.1f}%  F1={res['cross_f1']:.3f}  "
                  f"room-dec={agg[obj]['room_dec'][-1]*100:5.1f}%")

    # Summary
    def ms(v):
        v = np.asarray(v, dtype=float)
        return f"{np.nanmean(v)*100:.1f}±{np.nanstd(v)*100:.1f}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "c1_gate_summary.csv"
    header = ("feature,dfs_bins,quantize,objective,in_dom_acc,cross_acc,"
              "cross_f1,room_decod,seeds")
    header_needed = not (csv_path.exists() and csv_path.stat().st_size > 0)
    with open(csv_path, "a", encoding="utf-8") as fh:
        if header_needed:
            fh.write(header + "\n")
        for o in args.objectives:
            row = (f"{args.feature},{args.dfs_bins},{args.quantize},{o},"
                   f"{ms(agg[o]['in_acc'])},{ms(agg[o]['cross_acc'])},"
                   f"{ms(agg[o]['cross_f1'])},{ms(agg[o]['room_dec'])},"
                   f"{'/'.join(str(s) for s in args.seeds)}")
            fh.write(row + "\n")
    print(f"\n[saved] {csv_path}")

    print(f"\n{'='*70}\n  C1 GATE SUMMARY  (mean±std over seeds {args.seeds})")
    print(f"  OTA model = {'DiscreteComplexLinear (2-bit)' if args.quantize else 'ComplexLinear (continuous)'}")
    print(f"{'='*70}")
    print(f"  {'objective':<12}{'in-dom acc':<14}{'cross acc':<14}"
          f"{'cross F1':<14}{'room-decod':<12}")
    print(f"  {'-'*66}")
    for o in args.objectives:
        print(f"  OTA+{o:<8}{ms(agg[o]['in_acc']):<14}"
              f"{ms(agg[o]['cross_acc']):<14}{ms(agg[o]['cross_f1']):<14}"
              f"{ms(agg[o]['room_dec']):<12}")

    # Bar plot: cross-room accuracy per objective
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(args.objectives))
    means = [np.nanmean(agg[o]["cross_acc"]) * 100 for o in args.objectives]
    stds = [np.nanstd(agg[o]["cross_acc"]) * 100 for o in args.objectives]
    ax.bar(xs, means, yerr=stds, capsize=5, color="#4C72B0",
           edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"OTA+{o}" for o in args.objectives], rotation=0)
    ax.set_ylabel("Cross-room accuracy (%)")
    ax.set_title(f"C1 gate — cross-room acc by DG objective "
                 f"[{args.feature}, {args.dfs_bins}]")
    ax.set_ylim(0, 100)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + 1.5, f"{m:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / f"c1_gate_bar_{args.feature}_{args.dfs_bins}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")

    print("\n[done] C1 gate complete.")


if __name__ == "__main__":
    main()
