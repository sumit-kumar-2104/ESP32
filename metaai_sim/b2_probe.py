"""
B2 — Domain-predictability linear probe: probing and evaluation.

SUCCESS PATTERN (prediction P1):
  In `matched` mode (fair, equal-dimensionality comparison via PCA):
  - If the collapsing models (ota, mlp) retain MORE domain information than
    the generalizing Wi-CBR (which should be near chance), then P1 is supported:
    domain leakage in features explains the environment-specific collapse.
  - If ota, mlp, and wicbr all retain similar domain information, then
    feature-level leakage does NOT explain the collapse and the cause is the
    decision rule (Proposition 1 territory).

  The `raw` mode is confounded by dimensionality differences (ota=6, mlp=256,
  raw=8000, wicbr=1024). Higher-dimensional features trivially allow better
  linear separability. Only use `raw` for backward compatibility; `matched`
  mode is the fair comparison.

For each .npz in dumps/ and each domain factor (room, location, orientation,
user), this script evaluates:
  PRIMARY   — MLPClassifier (64 hidden) predicting domain
  SECONDARY — LogisticRegression predicting domain (linear probe)
  CONTROL   — LogisticRegression predicting gesture

Methodology:
  - StratifiedGroupKFold(n_splits=5) on groups; fallback StratifiedKFold
  - StandardScaler fit on train only
  - PCA fit on train only (matched mode), reducing all feature sets to K dims
  - 3 seeds × 5 folds → mean ± std of accuracy and macro-F1
  - Majority-class chance baseline (DummyClassifier) for both metrics

Requirements: numpy, scikit-learn, matplotlib
Usage:        python b2_probe.py
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# ─── Configuration ────────────────────────────────────────────────────────────

DUMPS_DIR = Path(__file__).parent / "dumps"
RESULTS_DIR = Path(__file__).parent / "results"
FEATURE_SETS = ["raw", "ota", "mlp", "wicbr"]
DOMAIN_FACTORS = ["room", "location", "orientation", "user"]
SEEDS = [42, 123, 7]
N_FOLDS = 5

# PCA target dimensions for matched mode
PCA_K_VALUES = [6, 32]
DEFAULT_PCA_K = 6  # matches smallest feature set (OTA = 6 class scores)

# Modes: "raw" = original (confounded by dimensionality), "matched" = PCA-equalized
PROBE_MODES = ["raw", "matched"]

# CSI support — when True, load CSI features instead of BVP-derived ones.
# CSI contains frequency-selective channel information and should show strong
# room/environment decodability because the channel IS the environment.
USE_CSI = False


def load_csi_features():
    """Load raw CSI features for domain probing.

    Raw CSI (Channel State Information) contains the frequency-selective
    wireless channel response. Unlike BVP (Body Velocity Profile), which is
    a gesture-level abstraction that removes environment information, CSI
    retains the full multipath structure of the room. Therefore:
      - Room/environment should be STRONGLY decodable from CSI (near 100%)
      - Location and orientation should also be highly decodable
      - This provides the upper bound for domain leakage

    When CSI files are available, this function should:
      1. Load CSI amplitude/phase matrices per sample
      2. Flatten or extract features (e.g., amplitude across subcarriers x packets)
      3. Return dict with keys: X, y_room, y_location, y_orientation, y_user,
         y_gesture, groups — same format as BVP-based .npz dumps

    Returns None if CSI data files are not present.
    """
    csi_path = DUMPS_DIR / "csi_raw.npz"
    if csi_path.exists():
        return np.load(csi_path)
    return None


# ─── Probing infrastructure ───────────────────────────────────────────────────

def _get_splits(X, y, groups, seed, n_folds=N_FOLDS):
    """Return CV splits using StratifiedGroupKFold or fallback."""
    if groups is not None and len(np.unique(groups)) >= n_folds:
        try:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return list(cv.split(X, y, groups))
        except ValueError:
            pass
    warnings.warn("groups absent or insufficient — using StratifiedKFold")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def run_probe(X, y, groups, clf_factory, pca_k=None):
    """Cross-validated probe → (acc_mean, acc_std, f1_mean, f1_std).

    If pca_k is set, fit PCA on training fold only and project both
    train and test to pca_k dimensions before classification.
    """
    accs, f1s = [], []
    for seed in SEEDS:
        for train_idx, test_idx in _get_splits(X, y, groups, seed):
            scaler = StandardScaler().fit(X[train_idx])
            X_tr = scaler.transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])

            if pca_k is not None and pca_k < X_tr.shape[1]:
                pca = PCA(n_components=pca_k, random_state=seed).fit(X_tr)
                X_tr = pca.transform(X_tr)
                X_te = pca.transform(X_te)

            clf = clf_factory(seed)
            clf.fit(X_tr, y[train_idx])
            y_pred = clf.predict(X_te)
            accs.append(accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def run_chance(X, y, groups):
    """Majority-class baseline → (acc_mean, acc_std, f1_mean, f1_std)."""
    accs, f1s = [], []
    for seed in SEEDS:
        for train_idx, test_idx in _get_splits(X, y, groups, seed):
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def logreg_factory(seed):
    return LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")


def mlp_factory(seed):
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed,
                         early_stopping=True, validation_fraction=0.15)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("B2 — Domain-predictability probe: evaluation")
    print("=" * 60)

    # Handle CSI mode
    if USE_CSI:
        csi_data = load_csi_features()
        if csi_data is not None:
            print("[CSI] Using CSI features for probing.")
        else:
            print("[CSI] USE_CSI=True but no CSI data found. Falling back to BVP.")

    # Load all available feature sets
    loaded = {}
    for name in FEATURE_SETS:
        npz_path = DUMPS_DIR / f"{name}.npz"
        if not npz_path.exists():
            print(f"[{name}] not found — skipping")
            continue
        loaded[name] = np.load(npz_path)
        print(f"[{name}] X.shape={loaded[name]['X'].shape}")

    if not loaded:
        print("\nNo feature dumps found. Run b2_dump_features.py first.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect P1 summary data across all factors and modes
    p1_summary = {}  # {(factor, mode): {feat_name: mlp_acc}}

    for mode in PROBE_MODES:
        pca_k = DEFAULT_PCA_K if mode == "matched" else None
        mode_label = (f"matched (PCA K={pca_k})" if mode == "matched"
                      else "raw (confounded by dimensionality)")

        print(f"\n{'#'*70}")
        print(f"  MODE: {mode_label}")
        if mode == "matched":
            print(f"  NOTE: This is the FAIR comparison — all features projected to K={pca_k} dims.")
        else:
            print(f"  NOTE: Confounded by size (ota=6 vs mlp=256 vs raw=8000 vs wicbr=1024).")
            print(f"         Use 'matched' mode for valid domain-leakage comparison.")
        print(f"{'#'*70}")

        for factor in DOMAIN_FACTORS:
            y_key = f"y_{factor}"

            # Drop room factor from BVP charts — BVP removes environment info.
            # Keep in code for when CSI is used (where room IS the channel).
            if factor == "room" and not USE_CSI:
                print(f"\n  [skip] '{factor}' — uninformative for BVP (environment removed by "
                      f"BVP extraction). Will be relevant when USE_CSI=True.")
                continue

            print(f"\n{'='*70}")
            print(f"  DOMAIN FACTOR: {factor}  |  MODE: {mode}")
            print(f"{'='*70}")

            ref_data = next(iter(loaded.values()))
            if y_key not in ref_data:
                print(f"  Key '{y_key}' missing from dumps — skipping factor")
                continue

            y_domain = ref_data[y_key]
            y_gesture = ref_data["y_gesture"]
            groups = ref_data["groups"] if "groups" in ref_data else None
            n_classes = len(np.unique(y_domain))
            print(f"  {n_classes} domain classes, {len(y_domain)} samples")

            if n_classes < 2:
                print(f"  Only 1 class — domain probing is trivial, skipping")
                continue

            # Chance baselines
            ch_a, ch_as, ch_f, ch_fs = run_chance(ref_data["X"], y_domain, groups)
            ch_ga, ch_gas, ch_gf, ch_gfs = run_chance(ref_data["X"], y_gesture, groups)
            print(f"  Chance domain:  acc={ch_a*100:.1f}±{ch_as*100:.1f}%  "
                  f"F1={ch_f:.3f}±{ch_fs:.3f}")
            print(f"  Chance gesture: acc={ch_ga*100:.1f}±{ch_gas*100:.1f}%  "
                  f"F1={ch_gf:.3f}±{ch_gfs:.3f}")

            rows = []
            bar_data = []

            for name in FEATURE_SETS:
                if name not in loaded:
                    continue
                data = loaded[name]
                X = data["X"]
                y_d = data[y_key]
                y_g = data["y_gesture"]
                g = data["groups"] if "groups" in data else None

                # PRIMARY: MLP domain probe
                ma, ms, mf, mfs = run_probe(X, y_d, g, mlp_factory, pca_k=pca_k)
                # SECONDARY: linear domain probe
                la, ls, lf, lfs = run_probe(X, y_d, g, logreg_factory, pca_k=pca_k)
                # CONTROL: linear gesture probe
                ga, gs, gf, gfs = run_probe(X, y_g, g, logreg_factory, pca_k=pca_k)

                print(f"\n  [{name}] (dim={X.shape[1]}" +
                      (f" -> PCA {pca_k})" if pca_k else ")"))
                print(f"    Domain MLP:     acc={ma*100:.1f}±{ms*100:.1f}%  "
                      f"F1={mf:.3f}±{mfs:.3f}")
                print(f"    Domain linear:  acc={la*100:.1f}±{ls*100:.1f}%  "
                      f"F1={lf:.3f}±{lfs:.3f}")
                print(f"    Gesture ctrl:   acc={ga*100:.1f}±{gs*100:.1f}%  "
                      f"F1={gf:.3f}±{gfs:.3f}")

                rows.append({
                    "features": name,
                    "dom_mlp_acc": f"{ma*100:.1f}±{ms*100:.1f}",
                    "dom_mlp_f1": f"{mf:.3f}±{mfs:.3f}",
                    "dom_lin_acc": f"{la*100:.1f}±{ls*100:.1f}",
                    "gest_acc": f"{ga*100:.1f}±{gs*100:.1f}",
                    "chance_acc": f"{ch_a*100:.1f}±{ch_as*100:.1f}",
                    "chance_f1": f"{ch_f:.3f}±{ch_fs:.3f}",
                })
                bar_data.append((name, ma, ms))

                # Store for P1 summary
                p1_summary.setdefault((factor, mode), {})[name] = ma

            # ─── Summary table ────────────────────────────────────────────
            print(f"\n  {'─'*90}")
            print(f"  {'Feats':<7} {'Dom MLP acc':<14} {'Dom MLP F1':<13} "
                  f"{'Dom lin acc':<14} {'Gest ctrl':<14} {'Chance acc':<14} {'Chance F1':<12}")
            print(f"  {'─'*90}")
            for r in rows:
                print(f"  {r['features']:<7} {r['dom_mlp_acc']:<14} {r['dom_mlp_f1']:<13} "
                      f"{r['dom_lin_acc']:<14} {r['gest_acc']:<14} "
                      f"{r['chance_acc']:<14} {r['chance_f1']:<12}")
            print(f"  {'─'*90}")

            # ─── Bar chart ────────────────────────────────────────────────
            if bar_data:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                names_plot = [d[0] for d in bar_data]
                means = [d[1] * 100 for d in bar_data]
                stds = [d[2] * 100 for d in bar_data]
                x = np.arange(len(names_plot))
                bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0",
                              edgecolor="black", linewidth=0.5, zorder=3)
                ax.axhline(ch_a * 100, color="gray", linestyle="--", linewidth=1,
                           label=f"chance ({ch_a*100:.1f}%)", zorder=2)
                ax.set_xticks(x)
                ax.set_xticklabels(names_plot)
                ax.set_ylabel("Domain decoding accuracy (%)")
                title_suffix = f" [{mode}, K={pca_k}]" if pca_k else f" [{mode}]"
                ax.set_title(f"B2: MLP probe — {factor} decodability{title_suffix}")
                ax.legend(loc="upper right")
                ax.set_ylim(0, 105)
                for bar, m in zip(bars, means):
                    ax.annotate(f"{m:.1f}", xy=(bar.get_x() + bar.get_width() / 2,
                                bar.get_height()),
                                xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=9)
                plt.tight_layout()
                out_path = RESULTS_DIR / f"b2_{factor}_{mode}.png"
                fig.savefig(out_path, dpi=150)
                plt.close()
                print(f"\n  [saved] {out_path}")

    # ─── P1 DECISIVE CONTRAST SUMMARY ────────────────────────────────────────
    print(f"\n\n{'█'*70}")
    print(f"  P1 DECISIVE CONTRAST: Domain Decoding (MLP probe, matched dims)")
    print(f"  Question: Do collapsing models (ota, mlp) retain MORE domain")
    print(f"            than the generalizing model (wicbr)?")
    print(f"{'█'*70}")

    collapsing_models = ["ota", "mlp"]
    generalizing_models = ["wicbr"]
    target_mode = "matched"

    for factor in DOMAIN_FACTORS:
        if factor == "room" and not USE_CSI:
            continue
        key = (factor, target_mode)
        if key not in p1_summary:
            continue
        results = p1_summary[key]

        print(f"\n  Factor: {factor}")
        print(f"  {'─'*50}")

        col_accs = []
        gen_accs = []
        for name in FEATURE_SETS:
            if name not in results:
                continue
            acc = results[name]
            marker = ""
            if name in collapsing_models:
                col_accs.append(acc)
                marker = " [COLLAPSING]"
            elif name in generalizing_models:
                gen_accs.append(acc)
                marker = " [GENERALIZING]"
            print(f"    {name:<7}: {acc*100:.1f}%{marker}")

        # Verdict
        if col_accs and gen_accs:
            col_mean = np.mean(col_accs)
            gen_mean = np.mean(gen_accs)
            diff = (col_mean - gen_mean) * 100
            if diff > 5:
                verdict = (f"P1 SUPPORTED: collapsing models retain {diff:.1f}pp MORE domain "
                           f"than Wi-CBR -> domain leakage explains collapse.")
            elif diff < -5:
                verdict = (f"UNEXPECTED: generalizing model retains MORE domain ({-diff:.1f}pp). "
                           f"Review methodology.")
            else:
                verdict = (f"INCONCLUSIVE: similar domain retention (delta={diff:.1f}pp). "
                           f"Feature leakage does NOT explain collapse -> decision-rule cause "
                           f"(Proposition 1).")
            print(f"    -> {verdict}")
        else:
            missing = []
            if not col_accs:
                missing.extend(collapsing_models)
            if not gen_accs:
                missing.extend(generalizing_models)
            print(f"    -> Cannot compute contrast (missing: {missing}). "
                  f"Run train_wicbr.py + b2_dump_features.py first.")

    print(f"\n{'█'*70}")
    print("[done] B2 probe complete.")


if __name__ == "__main__":
    main()
