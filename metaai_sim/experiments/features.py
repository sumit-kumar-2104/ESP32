"""Feature materialisation from the raw CSI manifest.

Given a list of ``RecordingRecord`` s already produced by
``experiments.manifest``, build the (X, y, sample_ids, group_ids) numpy
arrays for a chosen feature mode. Reuses ``data.csi_loader`` for the
low-level per-receiver feature blocks so this stays in lockstep with the
rest of the repo.

Also provides a synthetic fallback ("SYNTHETIC_FIXTURE") for smoke tests
where no dataset is present; results built from synthetic data MUST be
labelled as such by the runner — they are never scientific findings.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.manifest import RecordingRecord


@dataclass
class FeatureBundle:
    X: np.ndarray                 # (N, D)
    y: np.ndarray                 # (N,) 0-based model-class index
    sample_ids: list[str]
    group_ids: list[str]
    gestures_raw: list[int]       # canonical 1-based gesture id per sample
    class_index_to_gesture: dict[int, int]
    feature_mode: str
    dfs_bins: str
    is_synthetic: bool = False


def _class_map(gestures: Sequence[int]) -> tuple[np.ndarray, dict[int, int]]:
    uniq = sorted(set(gestures))
    remap = {g: i for i, g in enumerate(uniq)}
    y = np.array([remap[g] for g in gestures], dtype=np.int64)
    class_index_to_gesture = {i: g for g, i in remap.items()}
    return y, class_index_to_gesture


def build_features_from_manifest(
    records: Sequence[RecordingRecord],
    feature_mode: str,
    *,
    dfs_bins: str = "full",
    verbose: bool = False,
) -> FeatureBundle:
    """Build feature vectors by directly calling the existing csi_loader.

    We do NOT re-walk the disk. We iterate the manifest, ask ``csi_loader``
    for the per-receiver feature for each recorded .dat path, and assemble
    the fixed-length vector. Missing receivers stay zero.
    """
    from data.csi_loader import (
        _receiver_feature, feature_dim, N_RX_FILES,
    )
    per_rx, dim = feature_dim(feature_mode, dfs_bins=dfs_bins)
    X = np.zeros((len(records), dim), dtype=np.float32)
    sample_ids: list[str] = []
    group_ids: list[str] = []
    gestures: list[int] = []
    for i, r in enumerate(records):
        for rx_key, path in r.file_paths.items():
            try:
                rx = int(rx_key[1:])
            except ValueError:
                continue
            if not (1 <= rx <= N_RX_FILES):
                continue
            fr = _receiver_feature(path, feature_mode, dfs_bins=dfs_bins)
            if fr is None:
                continue
            X[i, (rx - 1) * per_rx: rx * per_rx] = fr
        sample_ids.append(r.sample_id)
        group_ids.append(r.group_id)
        gestures.append(r.gesture)
        if verbose and (i + 1) % 200 == 0:
            print(f"[features] {i + 1}/{len(records)} built")
    y, cmap = _class_map(gestures)
    return FeatureBundle(
        X=X, y=y, sample_ids=sample_ids, group_ids=group_ids,
        gestures_raw=gestures, class_index_to_gesture=cmap,
        feature_mode=feature_mode, dfs_bins=dfs_bins,
    )


def synthetic_fixture(
    n_recordings: int = 60,
    n_gestures: int = 3,
    dim: int = 32,
    seed: int = 42,
) -> tuple[list[RecordingRecord], FeatureBundle]:
    """Deterministic, clearly-labelled synthetic fixture for smoke runs.

    Produces manifest records with valid group_ids across two rooms/dates
    plus a fully separable feature bundle so the harness can be exercised
    end-to-end without a real dataset. Never a stand-in for research
    metrics: :attr:`FeatureBundle.is_synthetic` is set.
    """
    rng = np.random.default_rng(seed)
    records: list[RecordingRecord] = []
    gestures: list[int] = []
    X = np.zeros((n_recordings, dim), dtype=np.float32)
    for i in range(n_recordings):
        room = 1 if i < n_recordings // 2 else 2
        date = "20181109" if room == 1 else "20181118"
        user = 1 + (i % 3)
        gesture = 1 + (i % n_gestures)
        loc = 1 + (i % 2)
        ori = 1 + (i % 2)
        # rep must uniquely identify this recording within the fixture,
        # otherwise two rows collide on the (date, user, gesture, loc, ori,
        # rep) key and produce identical sample_ids.
        rep = 1 + i
        rec_id = f"{date}-u{user}-g{gesture}-l{loc}-o{ori}-r{rep}"
        group_id = f"{user}-{loc}-{ori}-{room}-{date}"
        sample_id = hashlib.sha256(rec_id.encode()).hexdigest()[:16]
        records.append(RecordingRecord(
            sample_id=sample_id, recording_id=rec_id, group_id=group_id,
            session_id=date, date=date, room=room, user=user,
            gesture=gesture, gesture_name=f"g{gesture}",
            location=loc, orientation=ori, repetition=rep,
            receivers_present=[1, 2, 3, 4, 5, 6],
            receivers_missing=[],
            file_paths={f"r{k}": f"synthetic/{rec_id}-r{k}.dat" for k in range(1, 7)},
            file_hashes={f"r{k}": hashlib.sha256(f"{rec_id}-{k}".encode()).hexdigest()[:16] for k in range(1, 7)},
            file_sizes={f"r{k}": 4096 for k in range(1, 7)},
            packet_counts={f"r{k}": 128 for k in range(1, 7)},
            quality="ok", quality_detail="synthetic",
        ))
        gestures.append(gesture)
        # Feature: gesture-specific centroid + Gaussian noise + tiny room offset.
        centroid = np.zeros(dim, dtype=np.float32)
        centroid[gesture % dim] = 5.0
        X[i] = centroid + rng.normal(0, 0.5, size=dim).astype(np.float32)
        X[i] += 0.1 * (room - 1.5)
    y, cmap = _class_map(gestures)
    bundle = FeatureBundle(
        X=X, y=y, sample_ids=[r.sample_id for r in records],
        group_ids=[r.group_id for r in records],
        gestures_raw=gestures, class_index_to_gesture=cmap,
        feature_mode="SYNTHETIC_FIXTURE", dfs_bins="n/a",
        is_synthetic=True,
    )
    return records, bundle


def slice_bundle(bundle: FeatureBundle, sample_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Slice a FeatureBundle down to the given sample_ids, preserving order."""
    id_to_row = {sid: i for i, sid in enumerate(bundle.sample_ids)}
    missing = [s for s in sample_ids if s not in id_to_row]
    if missing:
        raise KeyError(f"{len(missing)} sample_ids missing from feature bundle; first={missing[:3]}")
    rows = [id_to_row[s] for s in sample_ids]
    return bundle.X[rows], bundle.y[rows], list(sample_ids)
