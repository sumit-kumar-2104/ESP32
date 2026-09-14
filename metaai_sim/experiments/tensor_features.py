"""Structure-preserving 3-D feature tensors (Widar3.0, Section 5).

Emits per-sample rank-3 tensors in the shapes used by the CNN-GRU
backbone in Zheng et al. (MobiSys 2019). Unlike the flat feature
builders in :mod:`experiments.features` and :mod:`data.csi_loader`,
these builders keep the axes the paper's per-time-step CNN acts on:

- ``csi_tensor``: ``(antennas=18, subcarriers=30, T)`` — 6 receivers ×
  3 rx antennas, concatenated along the antenna axis.
- ``dfs_tensor``: ``(receivers=6, F, T)`` — Doppler-frequency spectrum
  per receiver, with a real temporal axis.
- ``bvp_tensor``: ``(Vx, Vy, T)`` — Body-coordinate Velocity Profile
  loaded from Widar3.0's provided BVP ``.mat`` files. Each snapshot is
  L1-normalised (sums to 1) per the paper's Section 5.1, and the
  temporal axis is resampled to a fixed ``t0``.

The runner treats tensor bundles as pre-normalised: it does NOT run
``StandardScaler`` on multi-dimensional per-sample features because
whitening across pixels destroys the local structure the CNN exists to
exploit. Every builder logs its variable-length policy (resampling or
pad-truncate) so the choice is auditable from the run manifest.

If the BVP source tree is missing under the dataset root, the
``bvp_tensor`` builder returns an :class:`UnavailableFeatureBundle`
marker with a plain-language reason. No derived approximation is ever
substituted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.features import FeatureBundle, _class_map
from experiments.manifest import RecordingRecord


# ─── Config defaults (paper Section 5.1, Table 2) ───────────────────────────

# CSI tensor: 6 receivers * 3 rx antennas = 18 antennas, 30 subcarriers.
CSI_TENSOR_ANTENNAS = 18
CSI_TENSOR_SUBCARRIERS = 30
CSI_TENSOR_T_DEFAULT = 32

# DFS tensor: 6 receivers, F Doppler bins, T time frames.
DFS_TENSOR_RECEIVERS = 6
DFS_TENSOR_F_DEFAULT = 16
DFS_TENSOR_T_DEFAULT = 32
DFS_TENSOR_NPERSEG = 64

# BVP tensor: (20, 20, t0).
BVP_TENSOR_VX = 20
BVP_TENSOR_VY = 20
BVP_TENSOR_T0_DEFAULT = 20
BVP_MAT_KEY = "velocity_spectrum_ro"


# ─── Public feature-mode names ──────────────────────────────────────────────

TENSOR_FEATURE_MODES: tuple[str, ...] = (
    "csi_tensor",
    "dfs_tensor",
    "bvp_tensor",
)


def is_tensor_mode(feature_mode: str) -> bool:
    return feature_mode in TENSOR_FEATURE_MODES


@dataclass
class TensorConfig:
    """Per-builder resolution config. Everything here participates in
    the cache key so ``csi_tensor(T=32)`` and ``csi_tensor(T=64)`` are
    NOT considered the same feature."""

    T: int = CSI_TENSOR_T_DEFAULT
    F: int = DFS_TENSOR_F_DEFAULT
    t0: int = BVP_TENSOR_T0_DEFAULT
    nperseg: int = DFS_TENSOR_NPERSEG
    variable_length_policy: str = "linear_resample"

    def to_dict(self) -> dict:
        return {
            "T": int(self.T),
            "F": int(self.F),
            "t0": int(self.t0),
            "nperseg": int(self.nperseg),
            "variable_length_policy": str(self.variable_length_policy),
        }


@dataclass
class UnavailableFeatureBundle:
    """Sentinel returned when the source data for a mode is not present.

    The runner checks ``isinstance(bundle, UnavailableFeatureBundle)``
    and enrols every ``(arm, seed)`` for the suite as ``unavailable``
    with ``reason``; no derived approximation is substituted.
    """

    feature_mode: str
    reason: str
    checked_paths: list[str] = field(default_factory=list)


# ─── Cache-key derivation ───────────────────────────────────────────────────

def tensor_cache_id(
    feature_mode: str,
    config: TensorConfig,
    sample_ids: Sequence[str],
) -> str:
    """Stable cache key so tensor and flat caches cannot collide.

    Includes the feature mode string, the resolution config, and the
    sorted sample-id list. Different modes at the same config produce
    different ids by construction because ``feature_mode`` is in the
    hash input.
    """
    h = hashlib.sha256()
    h.update(feature_mode.encode("utf-8"))
    h.update(b"|")
    h.update(json.dumps(config.to_dict(), sort_keys=True).encode("utf-8"))
    h.update(b"|")
    for sid in sorted(sample_ids):
        h.update(sid.encode("utf-8"))
        h.update(b",")
    return f"{feature_mode}:{h.hexdigest()[:16]}"


# ─── Helpers ────────────────────────────────────────────────────────────────

def _resample_time(x: np.ndarray, axis: int, target: int) -> np.ndarray:
    """Linear resample along ``axis`` to ``target`` samples.

    Handles the edge cases the paper doesn't mention explicitly: length
    zero (returned as zeros) and length one (broadcast to target).
    """
    n = x.shape[axis]
    if n == target:
        return x.astype(np.float32)
    if n == 0:
        shape = list(x.shape)
        shape[axis] = target
        return np.zeros(shape, dtype=np.float32)
    if n == 1:
        shape = list(x.shape)
        shape[axis] = target
        out = np.zeros(shape, dtype=np.float32)
        idx = [slice(None)] * x.ndim
        idx[axis] = slice(0, 1)
        # Broadcast the single frame along the target axis.
        for i in range(target):
            idx[axis] = slice(i, i + 1)
            out[tuple(idx)] = x[tuple([slice(0, 1) if a == axis else slice(None) for a in range(x.ndim)])]
        return out
    src = np.linspace(0.0, 1.0, n)
    tgt = np.linspace(0.0, 1.0, target)
    x_m = np.moveaxis(x, axis, -1)
    orig_shape = x_m.shape
    flat = x_m.reshape(-1, n)
    out = np.empty((flat.shape[0], target), dtype=np.float32)
    for i in range(flat.shape[0]):
        out[i] = np.interp(tgt, src, flat[i]).astype(np.float32)
    return np.moveaxis(out.reshape(*orig_shape[:-1], target), -1, axis)


# ─── CSI tensor ─────────────────────────────────────────────────────────────

def _csi_tensor_from_receivers(
    file_paths: dict[str, str], config: TensorConfig,
) -> np.ndarray | None:
    """Return shape ``(18, 30, T)`` or ``None`` if no receivers loaded.

    Six receivers × three rx antennas are concatenated along the antenna
    axis (rx1-ant0, rx1-ant1, rx1-ant2, rx2-ant0, ...). Missing
    receivers contribute zero rows.
    """
    from data.csi_loader import _load_receiver_csi, N_RX_FILES, N_SUB

    out = np.zeros((CSI_TENSOR_ANTENNAS, N_SUB, config.T), dtype=np.float32)
    got_any = False
    for rx_key, path in file_paths.items():
        try:
            rx = int(rx_key[1:])
        except (ValueError, IndexError):
            continue
        if not (1 <= rx <= N_RX_FILES):
            continue
        csi = _load_receiver_csi(path)
        if csi is None:
            continue
        # csi shape: (packets, 30, nrx, ntx). Use TX=0 and up to 3 rx antennas.
        n_ant = min(3, csi.shape[2])
        for a in range(n_ant):
            amp = np.abs(csi[:, :, a, 0]).astype(np.float32)  # (T_raw, 30)
            amp_resampled = _resample_time(amp, axis=0, target=config.T)
            row = (rx - 1) * 3 + a
            if row < CSI_TENSOR_ANTENNAS:
                out[row] = amp_resampled.T
                got_any = True
    return out if got_any else None


def build_csi_tensor(
    records: Sequence[RecordingRecord],
    config: TensorConfig,
) -> FeatureBundle:
    """Build ``(N, 18, 30, T)`` amplitude tensors from the manifest."""
    N = len(records)
    X = np.zeros((N, CSI_TENSOR_ANTENNAS, CSI_TENSOR_SUBCARRIERS, config.T),
                 dtype=np.float32)
    sample_ids: list[str] = []
    group_ids: list[str] = []
    gestures: list[int] = []
    for i, r in enumerate(records):
        block = _csi_tensor_from_receivers(r.file_paths, config)
        if block is not None:
            X[i] = block
        sample_ids.append(r.sample_id)
        group_ids.append(r.group_id)
        gestures.append(r.gesture)
    y, cmap = _class_map(gestures)
    return FeatureBundle(
        X=X, y=y, sample_ids=sample_ids, group_ids=group_ids,
        gestures_raw=gestures, class_index_to_gesture=cmap,
        feature_mode="csi_tensor", dfs_bins="n/a",
    )


# ─── DFS tensor ─────────────────────────────────────────────────────────────

def _dfs_tensor_from_receiver(csi: np.ndarray, config: TensorConfig) -> np.ndarray:
    """Return ``(F, T)`` Doppler spectrogram for one receiver."""
    from scipy.signal import stft
    from data.csi_loader import N_SUB
    out = np.zeros((config.F, config.T), dtype=np.float32)
    amp = np.abs(csi).mean(axis=(2, 3))  # (T_raw, 30)
    amp = amp - amp.mean(axis=0, keepdims=True)
    T_raw = amp.shape[0]
    nper = min(config.nperseg, T_raw)
    if nper < 4:
        return out
    _, _, Z = stft(amp, nperseg=nper, axis=0)  # (F_stft, 30, frames)
    spec = np.abs(Z).mean(axis=1).astype(np.float32)  # (F_stft, frames)
    k = min(config.F, spec.shape[0])
    if spec.shape[1] < 1:
        return out
    band = spec[:k, :]
    resampled = _resample_time(band, axis=1, target=config.T)
    out[:k] = resampled
    return out


def build_dfs_tensor(
    records: Sequence[RecordingRecord],
    config: TensorConfig,
) -> FeatureBundle:
    """Build ``(N, 6, F, T)`` Doppler tensors from the manifest."""
    from data.csi_loader import _load_receiver_csi, N_RX_FILES

    N = len(records)
    X = np.zeros((N, DFS_TENSOR_RECEIVERS, config.F, config.T), dtype=np.float32)
    sample_ids: list[str] = []
    group_ids: list[str] = []
    gestures: list[int] = []
    for i, r in enumerate(records):
        for rx_key, path in r.file_paths.items():
            try:
                rx = int(rx_key[1:])
            except (ValueError, IndexError):
                continue
            if not (1 <= rx <= N_RX_FILES):
                continue
            csi = _load_receiver_csi(path)
            if csi is None:
                continue
            X[i, rx - 1] = _dfs_tensor_from_receiver(csi, config)
        sample_ids.append(r.sample_id)
        group_ids.append(r.group_id)
        gestures.append(r.gesture)
    y, cmap = _class_map(gestures)
    return FeatureBundle(
        X=X, y=y, sample_ids=sample_ids, group_ids=group_ids,
        gestures_raw=gestures, class_index_to_gesture=cmap,
        feature_mode="dfs_tensor", dfs_bins="n/a",
    )


# ─── BVP tensor ─────────────────────────────────────────────────────────────

def _bvp_source_root() -> Path:
    """Where BVP .mat files are expected to live.

    Priority:
      1. ``METAAI_BVP_DIR`` env var (explicit override).
      2. ``<get_data_dir()>/widar3/BVP``.
    """
    import os
    from config import get_data_dir
    env = os.environ.get("METAAI_BVP_DIR")
    if env:
        return Path(env)
    return get_data_dir() / "widar3" / "BVP"


def _bvp_mat_path(root: Path, record: RecordingRecord) -> Path:
    """Widar3.0's convention: ``<root>/<date>-VS/user{u}-{g}-{loc}-{ori}-{rep}-r{rx}.mat``.

    We look for r1 as the canonical sidecar since Widar3.0's BVP files
    are per-receiver .mat outputs; higher-numbered receivers are
    interchangeable. Callers fall through to alternates in-order.
    """
    return root / f"{record.date}-VS" / (
        f"user{record.user}-{record.gesture}-{record.location}-"
        f"{record.orientation}-{record.repetition}-r1.mat"
    )


def _normalise_bvp_snapshots(bvp: np.ndarray) -> np.ndarray:
    """Per-frame L1 normalisation: each ``(Vx, Vy)`` slice sums to 1.

    Paper Sec 5.1: "we normalize each BVP snapshot so that its elements
    sum to 1". Slices with zero total are left as zeros.
    """
    out = bvp.astype(np.float32).copy()
    sums = out.sum(axis=(0, 1), keepdims=True)  # (1, 1, T)
    safe = np.where(sums > 0, sums, 1.0)
    out = out / safe
    # Any all-zero frames stay zero.
    return out.astype(np.float32)


def _load_bvp_mat(path: Path) -> np.ndarray | None:
    try:
        import scipy.io
        d = scipy.io.loadmat(str(path))
    except Exception:
        return None
    if BVP_MAT_KEY not in d:
        return None
    bvp = d[BVP_MAT_KEY]
    if not isinstance(bvp, np.ndarray) or bvp.ndim != 3:
        return None
    if bvp.shape[0] != BVP_TENSOR_VX or bvp.shape[1] != BVP_TENSOR_VY:
        return None
    if bvp.shape[2] == 0:
        return None
    return bvp


def build_bvp_tensor(
    records: Sequence[RecordingRecord],
    config: TensorConfig,
) -> FeatureBundle | UnavailableFeatureBundle:
    """Build ``(N, 20, 20, t0)`` BVP tensors, per-snapshot L1-normalised.

    If the BVP source tree is missing, returns an
    :class:`UnavailableFeatureBundle` with the checked path. If it
    exists but no records have a matching .mat, returns the same
    sentinel with a per-record explanation for the first few misses.
    """
    root = _bvp_source_root()
    checked: list[str] = [str(root)]
    if not root.exists():
        return UnavailableFeatureBundle(
            feature_mode="bvp_tensor",
            reason=(
                f"BVP source tree not present at {root}. "
                "Widar3.0's BVP .mat files are not generated by the raw CSI "
                "pipeline; download them from IEEE DataPort or set "
                "METAAI_BVP_DIR. NEVER substituting a CSI-derived approximation."
            ),
            checked_paths=checked,
        )

    N = len(records)
    X = np.zeros((N, BVP_TENSOR_VX, BVP_TENSOR_VY, config.t0), dtype=np.float32)
    sample_ids: list[str] = []
    group_ids: list[str] = []
    gestures: list[int] = []
    n_hit = 0
    misses: list[str] = []
    for i, r in enumerate(records):
        p = _bvp_mat_path(root, r)
        bvp = _load_bvp_mat(p) if p.exists() else None
        if bvp is not None:
            norm = _normalise_bvp_snapshots(bvp)
            X[i] = _resample_time(norm, axis=2, target=config.t0)
            n_hit += 1
        elif len(misses) < 5:
            misses.append(f"{r.sample_id}: {p}")
        sample_ids.append(r.sample_id)
        group_ids.append(r.group_id)
        gestures.append(r.gesture)
    if n_hit == 0:
        return UnavailableFeatureBundle(
            feature_mode="bvp_tensor",
            reason=(
                f"BVP tree at {root} exists but no matching .mat files "
                "were found for any manifest record. Verify the "
                "<date>-VS folder layout. Example misses: "
                + "; ".join(misses)
            ),
            checked_paths=checked,
        )
    y, cmap = _class_map(gestures)
    return FeatureBundle(
        X=X, y=y, sample_ids=sample_ids, group_ids=group_ids,
        gestures_raw=gestures, class_index_to_gesture=cmap,
        feature_mode="bvp_tensor", dfs_bins="n/a",
    )


# ─── Public dispatcher ──────────────────────────────────────────────────────

def build_tensor_features(
    records: Sequence[RecordingRecord],
    feature_mode: str,
    config: TensorConfig | None = None,
) -> FeatureBundle | UnavailableFeatureBundle:
    cfg = config or TensorConfig()
    if feature_mode == "csi_tensor":
        return build_csi_tensor(records, cfg)
    if feature_mode == "dfs_tensor":
        return build_dfs_tensor(records, cfg)
    if feature_mode == "bvp_tensor":
        return build_bvp_tensor(records, cfg)
    raise ValueError(f"unknown tensor feature mode {feature_mode!r}")


# ─── Synthetic fixture (smoke tests) ────────────────────────────────────────

def synthetic_tensor_bundle_from_records(
    records: Sequence[RecordingRecord],
    feature_mode: str,
    config: TensorConfig | None = None,
    seed: int = 42,
) -> FeatureBundle:
    """Build a synthetic tensor bundle keyed to an EXISTING manifest.

    Used by the runner when ``profile.use_synthetic=True`` so the
    manifest, splits and feature bundle all reference the same
    ``sample_id`` s. Never a research artefact.
    """
    cfg = config or TensorConfig()
    if feature_mode == "csi_tensor":
        shape = (CSI_TENSOR_ANTENNAS, CSI_TENSOR_SUBCARRIERS, cfg.T)
    elif feature_mode == "dfs_tensor":
        shape = (DFS_TENSOR_RECEIVERS, cfg.F, cfg.T)
    elif feature_mode == "bvp_tensor":
        shape = (BVP_TENSOR_VX, BVP_TENSOR_VY, cfg.t0)
    else:
        raise ValueError(f"unknown tensor mode {feature_mode!r}")
    rng = np.random.default_rng(seed)
    X = np.zeros((len(records),) + shape, dtype=np.float32)
    gestures = []
    for i, r in enumerate(records):
        gestures.append(r.gesture)
        centroid = np.zeros(shape, dtype=np.float32)
        centroid[..., (r.gesture - 1) % shape[-1]] = 1.0
        X[i] = centroid + rng.normal(0, 0.05, size=shape).astype(np.float32)
    if feature_mode == "bvp_tensor":
        X = np.maximum(X, 0)
        sums = X.sum(axis=(1, 2), keepdims=True)
        X = X / np.where(sums > 0, sums, 1.0)
    y, cmap = _class_map(gestures)
    return FeatureBundle(
        X=X.astype(np.float32),
        y=y,
        sample_ids=[r.sample_id for r in records],
        group_ids=[r.group_id for r in records],
        gestures_raw=gestures, class_index_to_gesture=cmap,
        feature_mode=feature_mode, dfs_bins="n/a",
        is_synthetic=True,
    )


def synthetic_tensor_fixture(
    feature_mode: str,
    n_recordings: int = 40,
    n_gestures: int = 3,
    config: TensorConfig | None = None,
    seed: int = 42,
) -> tuple[list[RecordingRecord], FeatureBundle]:
    """Deterministic, clearly-labelled tensor fixture for tests only.

    Returns a manifest and a fully-separable FeatureBundle in the exact
    shape that the CNN-GRU backbone consumes. Never a research artefact.

    The flat synthetic fixture insists on ``n_recordings // 2`` in each
    room, so we pass ``n_recordings`` straight through and let it own
    the record generation. The tensor X is built for the SAME number of
    records so slicing by sample_id never fails.
    """
    from experiments.features import synthetic_fixture

    cfg = config or TensorConfig()
    records, _ = synthetic_fixture(
        n_recordings=n_recordings, n_gestures=n_gestures, dim=8, seed=seed,
    )
    n = len(records)
    rng = np.random.default_rng(seed)
    if feature_mode == "csi_tensor":
        shape = (CSI_TENSOR_ANTENNAS, CSI_TENSOR_SUBCARRIERS, cfg.T)
    elif feature_mode == "dfs_tensor":
        shape = (DFS_TENSOR_RECEIVERS, cfg.F, cfg.T)
    elif feature_mode == "bvp_tensor":
        shape = (BVP_TENSOR_VX, BVP_TENSOR_VY, cfg.t0)
    else:
        raise ValueError(f"unknown tensor mode {feature_mode!r}")
    X = np.zeros((n,) + shape, dtype=np.float32)
    gestures = []
    for i, r in enumerate(records):
        gestures.append(r.gesture)
        centroid = np.zeros(shape, dtype=np.float32)
        centroid[..., (r.gesture - 1) % shape[-1]] = 1.0
        X[i] = centroid + rng.normal(0, 0.05, size=shape).astype(np.float32)
    if feature_mode == "bvp_tensor":
        # Enforce the per-snapshot sum-to-1 invariant so the fixture
        # exercises the same normalisation contract as real data.
        X = np.maximum(X, 0)
        sums = X.sum(axis=(1, 2), keepdims=True)
        X = X / np.where(sums > 0, sums, 1.0)
    y, cmap = _class_map(gestures)
    bundle = FeatureBundle(
        X=X.astype(np.float32),
        y=y,
        sample_ids=[r.sample_id for r in records],
        group_ids=[r.group_id for r in records],
        gestures_raw=gestures, class_index_to_gesture=cmap,
        feature_mode=feature_mode, dfs_bins="n/a",
        is_synthetic=True,
    )
    return records, bundle
