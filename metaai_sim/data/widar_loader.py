"""
Widar3.0 BVP data loader — reference-faithful, interpolation-aligned.

Pipeline (evidence-based from diagnostic analysis):
  1. Load (20,20,T) BVP from .mat (key 'velocity_spectrum_ro')
  2. Per-sample, per-time-frame min-max normalization to [0,1]
  3. Interpolate T axis to T_FIXED=20 (temporal alignment for linear model)
  4. Flatten to 8000-dim
  5. Random i.i.d. 90/10 stratified split, seed=42

Values stay in [0,1] so BPSK (2x-1) maps them to the full [-1,+1] range,
exactly like MNIST. Xavier weight init bounds the complex-model output.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io
import scipy.ndimage
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- Constants ----------------------------------------------------------------

NUM_GESTURES = 6
BVP_KEY = "velocity_spectrum_ro"
T_FIXED = 20  # interpolation target

VALID_DATES = [
    "20181109-VS",
    "20181115-VS",
    "20181117-VS",
    "20181118-VS",
    "20181121-VS",
    "20181127-VS",
    "20181128-VS",
    "20181130-VS",
    "20181204-VS",
    "20181205-VS",
    "20181208-VS",
    "20181209-VS",
    "20181211-VS",
]

IN_DOMAIN_DATE = "20181109-VS"

BUGGY_STEMS = {
    "user2-6-4-4-2-r1",
    "user3-1-3-1-8-r5",
    "user2-3-5-3-4-r4",
    "user6-3-1-1-5-r5",
    "user8-1-1-1-1-r5",
    "user8-3-3-3-5-r2",
    "user9-1-1-1-1-r1",
}


def _get_widar_dir() -> Path:
    from config import get_data_dir
    return get_data_dir() / "widar3"


def _is_buggy(stem: str) -> bool:
    for pattern in BUGGY_STEMS:
        if stem.startswith(pattern):
            return True
    return False


def _normalize_per_frame(data: np.ndarray) -> np.ndarray:
    """
    Per-sample, per-time-frame min-max normalization to [0,1].
    Matches reference widar3_keras.py normalize_data() exactly.
    """
    data_max = np.concatenate(
        (data.max(axis=0), data.max(axis=1)), axis=0
    ).max(axis=0)
    data_min = np.concatenate(
        (data.min(axis=0), data.min(axis=1)), axis=0
    ).min(axis=0)
    denom = data_max - data_min
    if len(np.where(denom == 0)[0]) > 0:
        return data
    data_max_rep = np.tile(data_max, (data.shape[0], data.shape[1], 1))
    data_min_rep = np.tile(data_min, (data.shape[0], data.shape[1], 1))
    return (data - data_min_rep) / (data_max_rep - data_min_rep)


def _interpolate_time(bvp: np.ndarray, t_target: int) -> np.ndarray:
    """Resample (20,20,T) -> (20,20,t_target) via linear interpolation."""
    t = bvp.shape[2]
    if t == t_target:
        return bvp
    zoom_factor = (1.0, 1.0, t_target / t)
    return scipy.ndimage.zoom(bvp, zoom_factor, order=1)


def get_widar_loaders(
    batch_size: int, seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load Widar3.0 BVP data with interpolation-aligned preprocessing.

    Returns:
        (train_loader, test_loader, feature_dim)
    """
    from config import WIDAR_DATE_SCOPE

    widar_dir = _get_widar_dir()
    bvp_root = widar_dir / "BVP"

    if not bvp_root.exists():
        print(f"ERROR: Widar3.0 BVP data not found at {bvp_root}")
        sys.exit(1)

    if WIDAR_DATE_SCOPE == "single":
        dates_to_use = [IN_DOMAIN_DATE]
        scope_label = f"single-date ({IN_DOMAIN_DATE}, in-domain, matches paper)"
    else:
        dates_to_use = VALID_DATES
        scope_label = f"pooled (all {len(VALID_DATES)} valid dates, cross-domain)"

    print(f"[data] Date scope: {scope_label}")
    print(f"[data] T-handling: interpolate to T={T_FIXED} (temporal alignment)")
    print(f"[data] Normalization: per-frame min-max [0,1] (BPSK maps to full [-1,+1])")
    print(f"[data] Loading BVP files...")

    features_list = []
    labels_list = []
    reps_list = []
    n_skipped_buggy = 0
    n_skipped_gesture = 0
    n_failed = 0

    for date_folder in dates_to_use:
        date_path = bvp_root / date_folder
        if not date_path.exists():
            continue
        for mat_file in sorted(date_path.rglob("*.mat")):
            stem = mat_file.stem
            if _is_buggy(stem):
                n_skipped_buggy += 1
                continue
            parts = stem.split("-")
            if len(parts) < 5:
                continue
            try:
                gesture_id = int(parts[1])
            except ValueError:
                continue
            if gesture_id < 1 or gesture_id > NUM_GESTURES:
                n_skipped_gesture += 1
                continue

            try:
                mat_data = scipy.io.loadmat(str(mat_file))
                if BVP_KEY not in mat_data:
                    n_failed += 1
                    continue
                bvp = mat_data[BVP_KEY]
                if bvp.ndim != 3 or bvp.shape[0] != 20 or bvp.shape[1] != 20:
                    n_failed += 1
                    continue
                if bvp.shape[2] == 0:
                    n_failed += 1
                    continue
            except Exception:
                n_failed += 1
                continue

            bvp_norm = _normalize_per_frame(bvp)
            bvp_interp = _interpolate_time(bvp_norm, T_FIXED)
            feature = bvp_interp.flatten().astype(np.float32)
            features_list.append(feature)
            labels_list.append(gesture_id - 1)
            try:
                rep_id = int(parts[4])
            except (ValueError, IndexError):
                rep_id = 1
            reps_list.append(rep_id)

    n_total = len(features_list)
    if n_total == 0:
        print("ERROR: No valid BVP files loaded.")
        sys.exit(1)

    features = np.stack(features_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)
    reps = np.array(reps_list, dtype=np.int64)

    print(f"[data] Loaded: {n_total} samples, {features.shape[1]}-dim raw")
    print(f"[data] Skipped: {n_skipped_buggy} buggy, "
          f"{n_skipped_gesture} non-canonical, {n_failed} failed/empty")

    from collections import Counter
    class_counts = Counter(labels.tolist())
    print(f"[data] Per-class counts: {dict(sorted(class_counts.items()))}")

    # Split: iid (random 90/10) or rep (reps 1-16 train, 17-20 test)
    from config import WIDAR_SPLIT
    if WIDAR_SPLIT == "rep":
        train_mask = reps <= 16
        test_mask = reps >= 17
        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        split_label = "rep-based (reps 1-16 train, 17-20 test)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.1, random_state=seed, stratify=labels
        )
        split_label = f"random i.i.d. 90/10, stratified, seed={seed}"

    feature_dim = features.shape[1]
    print(f"[data] Split: train={len(X_train)}, test={len(X_test)} ({split_label})")
    print(f"[data] INPUT_DIM = {feature_dim}")

    # Convert to tensors: (N, 1, feature_dim) for BPSK encoding
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader, feature_dim
