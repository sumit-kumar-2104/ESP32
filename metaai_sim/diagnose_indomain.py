"""
Phase 0 diagnostic — figure out WHY the current server run collapses to chance
on in-domain gesture classification.

For a chosen feature set (`--features {dfs_full, dfs_small, bvp, csi}`), on a
SINGLE in-domain random split (train + test drawn from the SAME rooms), this
script runs five diagnostics and writes a single human-readable report to
`logs/diag_<features>_<timestamp>.log`:

    1. Data sanity              — shape, dtype, per-class counts, NaN/Inf,
                                  feature min/max/mean/std.
    2. Label-consistency check  — Widar3.0 gesture-id <-> name mapping differs
                                  by date. Verify all included dates map ids
                                  to the SAME canonical gestures; FAIL LOUDLY
                                  if any date overloads an id.
    3. Unbiased probes          — StandardScaler (fit on train only), then
                                  sklearn LogisticRegression + MLPClassifier.
                                  Test accuracy is the deciding number:
                                    ~70-90%  -> features+labels are FINE,
                                               bug is in the OTA model/training.
                                    ~20-25% -> features or labels are BROKEN.
    4. OTA forward-path trace   — push one batch through the actual OTA path
                                  (BPSK/complex encode -> complex linear ->
                                  |.|) and report per-stage min/max/mean/
                                  abs-mean. Flag stages that collapse toward
                                  ~0 or explode >~50x the input scale.
    5. Normalization audit      — did the pipeline actually apply per-frame
                                  min-max normalization AND train-only
                                  standardization? warn if not.

This script does NOT attempt any fix. Phase 1 (`fix_pipeline.py`) provides
the disabled-by-default fix hooks.

Usage:
    python diagnose_indomain.py --features dfs_full   [--dates 20181109 20181118]
    python diagnose_indomain.py --features dfs_small
    python diagnose_indomain.py --features bvp        [single date, in-domain]
    python diagnose_indomain.py --features csi

Every run:
    - prints its compute device at startup
    - respects `--seed` (default 42)
    - never fabricates data: missing files fail loudly with the exact fix
"""

import argparse
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

# Silence sklearn's noisy convergence warnings (we run at fixed epochs to
# keep the diagnostic reproducible; a convergence warning is not a bug).
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, print_device, set_seed, get_data_dir


VALID_FEATURES = ("dfs_full", "dfs_small", "bvp", "csi", "raw_csi")


# ─── Data loaders (one per feature family) ────────────────────────────────────

def _load_bvp(dates):
    """Return (X, y, meta) where meta has per-sample date + gesture_id."""
    from data.widar_loader import (
        _get_widar_dir, _is_buggy, _normalize_per_frame, _interpolate_time,
        NUM_GESTURES, BVP_KEY, T_FIXED,
    )
    import scipy.io

    bvp_root = _get_widar_dir() / "BVP"
    if not bvp_root.exists():
        print(f"\n[FATAL] BVP directory not found at {bvp_root}")
        print(f"        Set METAAI_DATA_DIR or place the Widar3.0 BVP tree there.")
        sys.exit(1)

    xs, ys, meta_dates, meta_gids = [], [], [], []
    n_skip_buggy = n_skip_gid = n_fail = 0
    used_dates = list(dates) if dates else ["20181109-VS"]
    normalized_ok = 0
    normalized_skipped = 0
    for d in used_dates:
        # Accept both "20181109" and "20181109-VS"
        candidates = [d, d + "-VS", d[:8] + "-VS"]
        date_path = None
        date_folder = None
        for c in candidates:
            p = bvp_root / c
            if p.exists():
                date_path = p
                date_folder = c
                break
        if date_path is None:
            print(f"[warn] BVP date folder not found: tried {candidates}")
            continue
        for mat_file in sorted(date_path.rglob("*.mat")):
            stem = mat_file.stem
            if _is_buggy(stem):
                n_skip_buggy += 1
                continue
            parts = stem.split("-")
            if len(parts) < 5:
                continue
            try:
                gid = int(parts[1])
            except ValueError:
                continue
            if gid < 1 or gid > NUM_GESTURES:
                n_skip_gid += 1
                continue
            try:
                m = scipy.io.loadmat(str(mat_file))
                if BVP_KEY not in m:
                    n_fail += 1
                    continue
                bvp = m[BVP_KEY]
                if bvp.ndim != 3 or bvp.shape[:2] != (20, 20) or bvp.shape[2] == 0:
                    n_fail += 1
                    continue
            except Exception:
                n_fail += 1
                continue
            # Per-frame min-max normalization (audit will confirm this ran).
            bvp_n = _normalize_per_frame(bvp)
            # Cheap check: did the normalization actually push values into [0,1]?
            if bvp_n.min() >= -1e-6 and bvp_n.max() <= 1 + 1e-6:
                normalized_ok += 1
            else:
                normalized_skipped += 1
            bvp_i = _interpolate_time(bvp_n, T_FIXED)
            xs.append(bvp_i.flatten().astype(np.float32))
            ys.append(gid - 1)
            meta_dates.append(date_folder)
            meta_gids.append(gid)

    if not xs:
        print(f"\n[FATAL] No BVP samples loaded for dates {used_dates}.")
        sys.exit(1)

    X = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    meta = {
        "date": np.array(meta_dates),
        "gesture_id": np.array(meta_gids, dtype=np.int64),
        "normalized_ok": normalized_ok,
        "normalized_skipped": normalized_skipped,
        "n_skip_buggy": n_skip_buggy,
        "n_skip_gid": n_skip_gid,
        "n_fail": n_fail,
    }
    return X, y, meta


def _load_csi(feature, dfs_bins, dates, users, gestures):
    """Return (X, y, meta) for a CSI feature mode."""
    from data.csi_loader import build_csi_features
    from config import get_raw_csi_dir, require_raw_csi_dir
    csi_root = get_raw_csi_dir()
    if not csi_root.exists() or not any(csi_root.iterdir()):
        # For the raw_csi feature the user explicitly asked for raw CSI —
        # print the definitive fix message and stop. Same message for the
        # DFS features so behaviour is consistent.
        require_raw_csi_dir()
    data = build_csi_features(
        csi_root, dates,
        keep_users=set(users) if users else None,
        keep_gestures=set(gestures) if gestures else None,
        feature=feature, dfs_bins=dfs_bins, verbose=True,
    )
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y_gesture"], dtype=np.int64)
    used_dates = list(dates)
    meta = {
        "date": np.array([used_dates[0]] * len(X)) if len(used_dates) == 1
                else np.array(["multi"] * len(X)),
        "gesture_id": (y + 1).astype(np.int64),
        "used_dates": used_dates,
    }
    return X, y, meta


def load_features(args):
    """Dispatch to the right loader and return (X, y, meta, description)."""
    if args.features == "bvp":
        # BVP: single in-domain date by default.
        dates = args.dates if args.dates else ["20181109-VS"]
        X, y, meta = _load_bvp(dates)
        return X, y, meta, f"BVP (per-frame min-max, T={20} interpolation); dates={dates}"
    if args.features == "csi":
        dates = args.dates if args.dates else ["20181109"]
        X, y, meta = _load_csi("amp", "full", dates, args.users, args.gestures)
        return X, y, meta, f"CSI amp (360-dim); dates={dates}"
    if args.features == "dfs_full":
        dates = args.dates if args.dates else ["20181109"]
        X, y, meta = _load_csi("dfs_spec", "full", dates, args.users, args.gestures)
        return X, y, meta, f"CSI dfs_spec FULL (1536-dim); dates={dates}"
    if args.features == "dfs_small":
        dates = args.dates if args.dates else ["20181109"]
        X, y, meta = _load_csi("dfs_spec", "small", dates, args.users, args.gestures)
        return X, y, meta, f"CSI dfs_spec SMALL (150-dim); dates={dates}"
    if args.features == "raw_csi":
        dates = args.dates if args.dates else ["20181109"]
        X, y, meta = _load_csi("raw", "full", dates, args.users, args.gestures)
        return X, y, meta, (f"CSI raw (subcarrier-resolved amplitude, "
                            f"time-resampled to 32 frames, 6 receivers -> "
                            f"5760-dim); dates={dates}")
    raise ValueError(args.features)


# ─── Diagnostics ──────────────────────────────────────────────────────────────

def diag_data_sanity(X, y):
    """1. Print shape, dtype, class dist, NaN/Inf, min/max/mean/std."""
    print("\n" + "=" * 70)
    print("  1. DATA SANITY")
    print("=" * 70)
    print(f"  X.shape = {X.shape}   dtype = {X.dtype}")
    print(f"  y.shape = {y.shape}   dtype = {y.dtype}")
    ctr = Counter(y.tolist())
    print(f"  Per-class counts: {dict(sorted(ctr.items()))}")
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    print(f"  NaN count = {n_nan}    Inf count = {n_inf}")
    assert n_nan == 0, f"FAIL: X contains {n_nan} NaNs — clean the pipeline first."
    assert n_inf == 0, f"FAIL: X contains {n_inf} Infs — clean the pipeline first."
    print(f"  X min={X.min():.4g}  max={X.max():.4g}  "
          f"mean={X.mean():.4g}  std={X.std():.4g}")
    print(f"  |X| mean = {np.abs(X).mean():.4g}   "
          f"per-sample L2 mean = {np.linalg.norm(X, axis=1).mean():.4g}")


def diag_label_consistency(meta):
    """2. Verify per-date gesture_id -> canonical name is consistent."""
    from gesture_map import (
        canonical_name, is_unverified, date_prefix, GESTURE_MAP_BY_DATE,
    )
    print("\n" + "=" * 70)
    print("  2. LABEL-CONSISTENCY CHECK")
    print("=" * 70)
    dates = meta.get("date")
    gids = meta.get("gesture_id")
    if dates is None or gids is None or len(dates) == 0:
        print("  [skip] no per-sample date/gesture_id metadata available.")
        return

    # (date_prefix, gid) -> canonical name; None if unknown
    seen = {}
    for d, g in zip(dates, gids):
        key = (date_prefix(d), int(g))
        if key in seen:
            continue
        seen[key] = canonical_name(d, int(g))

    # Group by gid: which name does each date give to this id?
    per_gid = {}
    for (d, g), name in seen.items():
        per_gid.setdefault(g, []).append((d, name))

    print(f"  {'gesture_id':<12}{'date':<12}{'resolved_name':<25}")
    print(f"  {'-'*49}")
    any_unknown = False
    any_overload = False
    for gid in sorted(per_gid.keys()):
        names_seen = set()
        for d, name in per_gid[gid]:
            disp = name if name is not None else "<UNKNOWN>"
            print(f"  {gid:<12}{d:<12}{disp:<25}")
            if name is None:
                any_unknown = True
            else:
                names_seen.add(name)
        if len(names_seen) > 1:
            any_overload = True
            print(f"    !! OVERLOAD: gesture_id={gid} resolves to "
                  f"multiple names: {sorted(names_seen)}")

    unverified = sorted({date_prefix(d) for d in dates
                         if is_unverified(d)})
    if unverified:
        print(f"\n  [note] Unverified date prefixes present: {unverified}")
        print(f"         Confirm their mapping in gesture_map.py before pooling.")

    if any_overload:
        print("\n  [FAIL] Gesture-id overload detected across the loaded dates.")
        print("         Pooling these dates without a per-date remap will silently")
        print("         mis-label samples. Restrict to a single date OR add a")
        print("         canonical remap step in the loader before proceeding.")
        # We continue so the rest of the report is still written, but exit non-zero.
        return "OVERLOAD"

    if any_unknown:
        print("\n  [WARN] At least one (date, gesture_id) pair has no mapping in")
        print("         gesture_map.py. Add it there or exclude the date.")
        return "UNKNOWN"

    print("\n  [OK] All included (date, gesture_id) pairs resolve to a single")
    print("       canonical gesture name — labels are consistent.")
    return "OK"


def diag_probes(X, y, seed):
    """3. StandardScaler + LogReg + MLP on a single in-domain 80/20 split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 70)
    print("  3. UNBIASED PROBES (LogisticRegression + MLPClassifier)")
    print("=" * 70)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_te_s = sc.transform(X_te)

    print(f"  train N = {len(X_tr)}   test N = {len(X_te)}   dim = {X.shape[1]}")

    lr = LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs",
                            multi_class="auto")
    lr.fit(X_tr_s, y_tr)
    lr_acc = accuracy_score(y_te, lr.predict(X_te_s))
    print(f"  LogisticRegression test acc = {lr_acc*100:.2f}%")

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                        random_state=seed, early_stopping=True,
                        validation_fraction=0.15)
    mlp.fit(X_tr_s, y_tr)
    mlp_acc = accuracy_score(y_te, mlp.predict(X_te_s))
    print(f"  MLPClassifier      test acc = {mlp_acc*100:.2f}%")

    n_classes = len(np.unique(y))
    chance = 100.0 / n_classes
    print(f"  chance level               = {chance:.2f}%")

    return lr_acc, mlp_acc, chance / 100.0


def diag_ota_trace(X, y, seed, device):
    """4. Push one batch through the OTA model and report per-stage stats."""
    import torch
    from models.linear_complex import ComplexLinear

    print("\n" + "=" * 70)
    print("  4. OTA FORWARD-PATH TRACE")
    print("=" * 70)
    input_dim = X.shape[1]
    num_classes = int(np.unique(y).size)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(64, len(X)), replace=False)
    x_batch = torch.tensor(X[idx], dtype=torch.float32, device=device)

    def stats(name, t):
        # Handle complex tensors by taking magnitude for min/max/mean of |.|.
        if torch.is_complex(t):
            mag = torch.abs(t)
            return (name, float(mag.min()), float(mag.max()),
                    float(mag.mean()), float(mag.mean()))
        return (name, float(t.min()), float(t.max()),
                float(t.mean()), float(t.abs().mean()))

    print(f"  batch shape = {tuple(x_batch.shape)}   input_dim = {input_dim}   "
          f"num_classes = {num_classes}")

    # Stage A: raw input
    rows = [stats("A raw input", x_batch)]

    # Stage B: BPSK / complex encoding.
    # The paper convention is bipolar = 2x - 1 when x is in [0,1]; here CSI
    # features may already be zero-centered after standardization, so we
    # apply the same operation the training loop actually uses (no rescale).
    bipolar = 2.0 * x_batch - 1.0
    x_complex = torch.complex(bipolar, torch.zeros_like(bipolar))
    rows.append(stats("B after BPSK encode (|.|)", x_complex))

    # Stage C: complex linear layer with Xavier init (matches train.py Widar).
    torch.manual_seed(seed)
    model = ComplexLinear(input_dim, num_classes).to(device)
    with torch.no_grad():
        y_complex = torch.matmul(x_complex, model.complex_weight)
    rows.append(stats("C after complex linear (|.|)", y_complex))

    # Stage D: magnitude / readout — the classifier's logits.
    with torch.no_grad():
        y_mag = torch.abs(y_complex)
    rows.append(stats("D after magnitude readout", y_mag))

    print(f"\n  {'stage':<35}{'min':>10}{'max':>10}{'mean':>10}{'|.|mean':>10}")
    print(f"  {'-'*75}")
    for name, mn, mx, me, am in rows:
        print(f"  {name:<35}{mn:>10.3g}{mx:>10.3g}{me:>10.3g}{am:>10.3g}")

    # Flag collapse / explosion by comparing |.| mean of every stage against
    # the input |.| mean. This catches both bug classes we hit before:
    #   - a normalization that crushes signal (>= 10x shrink)
    #   - a scale explosion on high-dim inputs (>= 50x growth)
    input_ref = max(rows[0][4], 1e-12)
    print("\n  Sanity checks vs. input scale:")
    flags = []
    for name, _, _, _, am in rows[1:]:
        ratio = am / input_ref
        if ratio < 0.02:
            msg = f"COLLAPSE ({ratio:.3g}x input) — normalization crushing signal?"
            flags.append((name, msg))
        elif ratio > 50.0:
            msg = f"EXPLOSION ({ratio:.3g}x input) — Xavier scale wrong for this dim?"
            flags.append((name, msg))
        else:
            msg = f"ok ({ratio:.3g}x input)"
        print(f"    {name:<35}{msg}")
    return flags


def diag_normalization_audit(args, X, meta):
    """5. Report whether per-frame normalization and train-only standardization
    are actually applied by the CURRENT pipeline that produced X."""
    print("\n" + "=" * 70)
    print("  5. NORMALIZATION AUDIT")
    print("=" * 70)
    warns = []

    if args.features == "bvp":
        ok = int(meta.get("normalized_ok", 0))
        skip = int(meta.get("normalized_skipped", 0))
        print(f"  per-frame min-max normalization applied: "
              f"{ok} samples in [0,1], {skip} samples out of range")
        if X.min() < -1e-6 or X.max() > 1 + 1e-6:
            warns.append("BVP features are NOT strictly in [0,1] — per-frame min-max "
                         "may have been skipped for a subset (constant-denominator "
                         "recordings).")
    else:
        # CSI feature modes are NOT range-normalized inside csi_loader (they
        # come out as raw amplitude / spectrogram magnitudes). Downstream
        # scripts (b5_isolation.py, this diagnostic) are expected to apply
        # StandardScaler on the training fold BEFORE feeding the model.
        print(f"  CSI feature raw range: min={X.min():.4g}  max={X.max():.4g}  "
              f"mean={X.mean():.4g}  std={X.std():.4g}")
        if abs(X.mean()) > 0.5 or X.std() > 5 or X.std() < 0.05:
            warns.append("CSI feature stats are far from zero-mean unit-std — "
                         "downstream training MUST fit StandardScaler on the "
                         "TRAIN fold only before feeding the OTA / MLP model. "
                         "Failure to do so is a known cause of the collapse.")

    # Confirm this diagnostic itself used train-only standardization (it did,
    # in diag_probes). This is a written note for the log.
    print("  This script's probes (Diagnostic 3) fit StandardScaler on the "
          "TRAIN fold only.")
    print("  If the production training loop does NOT do the same, that is a bug.")

    if warns:
        print("\n  [WARN]")
        for w in warns:
            print(f"    - {w}")
    else:
        print("\n  [OK] Normalization looks reasonable for this feature family.")


# ─── Interpretation ───────────────────────────────────────────────────────────

def interpret(lr_acc, mlp_acc, chance, flags, label_status):
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    good = max(lr_acc, mlp_acc)
    if good >= 0.60:
        print(f"  Best sklearn probe = {good*100:.1f}% (>> chance {chance*100:.1f}%).")
        print(f"  -> FEATURES + LABELS ARE FINE for in-domain classification.")
        print(f"  -> The bug is in the OTA model or its training loop.")
        print(f"     Focus on: BPSK/complex encoding, Xavier scale for the")
        print(f"     current input_dim, and the magnitude-readout gradient path.")
    elif good <= max(chance * 1.5, 0.30):
        print(f"  Best sklearn probe = {good*100:.1f}% (near chance {chance*100:.1f}%).")
        print(f"  -> FEATURES or LABELS are BROKEN. No model choice can recover.")
        print(f"     Focus on: label mapping (see Diagnostic 2), feature")
        print(f"     extraction pipeline, or per-frame normalization step.")
    else:
        print(f"  Best sklearn probe = {good*100:.1f}% (in the 30-60% grey zone).")
        print(f"  -> Weak signal in features; expect the OTA linear+|.| model")
        print(f"     to sit at chance because it cannot exploit weak nonlinear")
        print(f"     structure. Improve features or move to a stronger readout.")

    if label_status in ("OVERLOAD", "UNKNOWN"):
        print(f"\n  Label check status = {label_status}. Even if the probes look")
        print(f"  fine, some samples are being mislabeled — fix that first.")

    if flags:
        print(f"\n  OTA-trace flags:")
        for name, msg in flags:
            print(f"    {name}: {msg}")
        print(f"  -> These are the two bug classes that produced the current chance-")
        print(f"     level result. Phase 1 (fix_pipeline.py) has disabled-by-default")
        print(f"     hooks to address each: remove signal-crushing normalization,")
        print(f"     and rescale OTA outputs to match Xavier scale.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", choices=VALID_FEATURES, required=True)
    ap.add_argument("--dates", nargs="+", default=None,
                    help="dates to include; default = single in-domain date "
                         "(20181109 / 20181109-VS)")
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3],
                    help="CSI: users to keep (ignored for BVP)")
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6],
                    help="CSI: gesture ids to keep (ignored for BVP)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Log to logs/diag_<features>_<timestamp>.log
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(f"diag_{args.features}")
    print(f"[diag] Phase 0 diagnostic — features = {args.features}")
    print(f"[diag] log = {log_path}   timestamp = {ts}")

    device = print_device()
    set_seed(args.seed)

    X, y, meta, desc = load_features(args)
    print(f"[diag] {desc}")

    diag_data_sanity(X, y)
    label_status = diag_label_consistency(meta)
    lr_acc, mlp_acc, chance = diag_probes(X, y, args.seed)
    flags = diag_ota_trace(X, y, args.seed, __import__("torch").device(
        "cuda" if __import__("torch").cuda.is_available() else "cpu"))
    diag_normalization_audit(args, X, meta)
    interpret(lr_acc, mlp_acc, chance, flags, label_status)

    print("\n[done] Phase 0 diagnostic complete.")
    print(f"[done] Report saved to: {log_path}")


if __name__ == "__main__":
    main()
