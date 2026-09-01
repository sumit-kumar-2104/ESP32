"""
Widar3.0 raw CSI loader for the B2 domain-predictability probe.

Reads Intel 5300 .dat files (one per receiver r1..r6), builds one fixed-size
feature vector per gesture sample, and attaches domain metadata:
    room        -> from the date folder (IEEE DataPort README room map)
    user/gesture/location/orientation/rep -> parsed from the filename
        user{id}-{gesture}-{loc}-{ori}-{rep}-r{rx}.dat

Feature per sample (subcarrier-resolved, antenna-averaged, 6 receivers).
Selectable via the `feature` argument (default "amp" = original behaviour):

    amp (default, 360-dim):
        for each receiver r1..r6:
            amp = |CSI| averaged over antenna pairs   -> (T, 30)
            [ mean_t(amp) (30) , std_t(amp) (30) ]    -> 60
        -> 6 * 60 = 360

    amp_phase (720-dim):
        amp block (60) PLUS a sanitized-phase block per receiver:
            phase unwrapped across subcarriers, per-packet linear slope
            (STO/CFO) removed, then [ mean_t (30), std_t (30) ]   -> 60
        -> 6 * (60 + 60) = 720

    amp_dfs (456-dim):
        amp block (60) PLUS a compact Doppler block per receiver:
            STFT along the packet/time axis of the antenna-averaged
            amplitude (DC removed), magnitude averaged over subcarriers and
            time frames, first DFS_BINS (=16) low-frequency bins kept   -> 16
        -> 6 * (60 + 16) = 456

    dfs_spec (1536-dim, TEMPORAL AXIS PRESERVED):
        Doppler-frequency spectrogram per receiver — the temporal micro-
        Doppler signature that motion actually leaves on the channel. For
        each receiver:
            amp = |CSI| averaged over antenna pairs                 -> (T, 30)
            DC-detrend along time, then STFT along the packet/time
              axis PER subcarrier, magnitude, averaged over subcarriers
              -> (F, frames)
            keep the first DFS_SPEC_BINS (=16) low-Doppler bins    -> (16, frames)
            resample the time axis to a fixed DFS_SPEC_FRAMES (=16) length
              via linear interpolation (pad-then-interp when short)
              -> (16, 16), flattened row-major -> 256 per receiver
        -> 6 * (16 * 16) = 1536, time axis NOT collapsed.

Receivers are concatenated in fixed order r1..r6 (missing receiver = zeros).

The subcarrier axis is kept because the room signature is per-frequency (the
core B1 argument). Time-mean captures the static channel (room); time-std
captures motion energy (gesture, used by the control probe).
"""

import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

N_SUB = 30
N_RX_FILES = 6                      # receivers r1..r6

# ─ feature-mode geometry ─────────────────────────────────────────────────────
AMP_PER_RX = 2 * N_SUB              # mean_t + std_t of amplitude          -> 60
PHASE_PER_RX = 2 * N_SUB           # mean_t + std_t of sanitized phase     -> 60
DFS_BINS = 16                      # low-freq Doppler bins kept per receiver
DFS_NPERSEG = 64                   # STFT window length (packets)

# dfs_spec geometry — Doppler-frequency spectrogram with time preserved.
DFS_SPEC_BINS = 16                  # low-Doppler bins kept per receiver
DFS_SPEC_FRAMES = 16                # fixed number of STFT time frames
DFS_SPEC_PER_RX = DFS_SPEC_BINS * DFS_SPEC_FRAMES   # 256 per receiver

# `small` dfs_spec geometry — a compact low-Doppler regime so the total
# feature dim (~150) is small enough for a single linear+magnitude readout
# (OTA_linear / Digital_LinMag) to have a fair chance of fitting in-domain.
DFS_SPEC_BINS_SMALL = 5             # low-Doppler bins kept per receiver
DFS_SPEC_FRAMES_SMALL = 5           # STFT time frames kept per receiver
DFS_SPEC_PER_RX_SMALL = DFS_SPEC_BINS_SMALL * DFS_SPEC_FRAMES_SMALL   # 25/rx -> 150 total

DFS_BINS_MODES = ("full", "small")

FEATURE_MODES = ("amp", "amp_phase", "amp_dfs", "dfs_spec")


def _dfs_spec_geometry(dfs_bins: str = "full"):
    """Return (bins, frames, per_rx) for a dfs_spec size regime."""
    if dfs_bins not in DFS_BINS_MODES:
        raise ValueError(
            f"unknown dfs_bins mode {dfs_bins!r}; choose from {DFS_BINS_MODES}")
    if dfs_bins == "small":
        return DFS_SPEC_BINS_SMALL, DFS_SPEC_FRAMES_SMALL, DFS_SPEC_PER_RX_SMALL
    return DFS_SPEC_BINS, DFS_SPEC_FRAMES, DFS_SPEC_PER_RX

# Back-compat defaults (amp mode).
FEAT_PER_RX = AMP_PER_RX                 # 60
FEAT_DIM = N_RX_FILES * FEAT_PER_RX      # 360


def feature_dim(feature: str = "amp", dfs_bins: str = "full"):
    """Return (per_receiver_dim, total_dim) for a feature mode.

    `dfs_bins` only affects the dfs_spec mode; other modes are unchanged.
    """
    if feature not in FEATURE_MODES:
        raise ValueError(
            f"unknown feature mode {feature!r}; choose from {FEATURE_MODES}")
    if feature == "dfs_spec":
        _, _, per_rx = _dfs_spec_geometry(dfs_bins)
        return per_rx, per_rx * N_RX_FILES
    per = AMP_PER_RX
    if feature == "amp_phase":
        per += PHASE_PER_RX
    elif feature == "amp_dfs":
        per += DFS_BINS
    return per, per * N_RX_FILES

# date-folder prefix -> room number (from IEEE DataPort README)
DATE_ROOM = {
    "20181109": 1, "20181112": 1, "20181115": 1, "20181116": 1,
    "20181121": 1, "20181130": 1,
    "20181117": 2, "20181118": 2, "20181127": 2, "20181128": 2,
    "20181204": 2, "20181205": 2, "20181208": 2, "20181209": 2,
    "20181211": 3,
}

_FNAME_RE = re.compile(r"^user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)$")


def date_to_room(date_folder: str) -> int:
    return DATE_ROOM.get(date_folder[:8], 0)


def _load_receiver_csi(dat_path):
    """Read one .dat and return scaled CSI (count, 30, nrx, ntx), or None."""
    import csiread
    try:
        r = csiread.Intel(dat_path)
        r.read()
        csi = r.get_scaled_csi()          # (count, 30, nrx, ntx)
    except Exception:
        return None
    if csi is None or csi.ndim != 4 or csi.shape[0] == 0 or csi.shape[1] != N_SUB:
        return None
    return csi


def _amp_block(csi):
    """[mean_t, std_t] of antenna-averaged amplitude -> (60,)."""
    amp = np.abs(csi).mean(axis=(2, 3))   # (T, 30)
    mean_t = amp.mean(axis=0)             # (30,)
    std_t = amp.std(axis=0)              # (30,)
    return np.concatenate([mean_t, std_t])


def _phase_block(csi):
    """Sanitized-phase [mean_t, std_t] -> (60,).

    Sanitize = unwrap across subcarriers, then remove the per-packet linear
    phase slope (STO/CFO detrend) so only environment/motion-dependent phase
    curvature remains.
    """
    c = csi.mean(axis=(2, 3))                 # complex (T, 30) antenna-averaged
    ph = np.unwrap(np.angle(c), axis=1)       # unwrap across subcarriers
    idx = np.arange(N_SUB, dtype=np.float64)
    idx_c = idx - idx.mean()
    denom = (idx_c ** 2).sum()
    ph_mean = ph.mean(axis=1, keepdims=True)
    slope = ((ph - ph_mean) * idx_c).sum(axis=1, keepdims=True) / denom
    ph_det = ph - (slope * idx_c + ph_mean)   # remove per-packet linear slope
    mean_t = ph_det.mean(axis=0)             # (30,)
    std_t = ph_det.std(axis=0)              # (30,)
    return np.concatenate([mean_t, std_t])


def _dfs_block(csi):
    """Compact Doppler spectrum -> (DFS_BINS,).

    STFT along the packet/time axis of the antenna-averaged amplitude (DC
    removed), magnitude averaged over subcarriers and time frames, keeping the
    first DFS_BINS low-frequency bins (human motion is low-Doppler).
    """
    from scipy.signal import stft
    amp = np.abs(csi).mean(axis=(2, 3))          # (T, 30)
    amp = amp - amp.mean(axis=0, keepdims=True)  # remove static/DC channel
    T = amp.shape[0]
    band = np.zeros(DFS_BINS, dtype=np.float64)
    nper = min(DFS_NPERSEG, T)
    if nper < 4:
        return band
    _, _, Z = stft(amp, nperseg=nper, axis=0)    # (F, 30, frames)
    spec = np.abs(Z).mean(axis=(1, 2))           # (F,) avg over subcarriers+time
    k = min(DFS_BINS, spec.shape[0])
    band[:k] = spec[:k]
    return band


def _dfs_spec_block(csi, dfs_bins: str = "full"):
    """Doppler-frequency spectrogram, temporal structure preserved.

    STFT along the packet/time axis of the antenna-averaged amplitude (DC
    removed), magnitude averaged over subcarriers, keeping the first
    n_bins low-Doppler bins and linearly resampling the STFT time axis to
    n_frames. Output shape (n_bins, n_frames), row-major-flattened. Missing
    or short recordings are zero-padded.

    `dfs_bins`: "full" keeps the original 16x16 geometry; "small" keeps a
    compact low-Doppler band (5 bins x 5 frames = 25 per receiver).
    """
    from scipy.signal import stft
    n_bins, n_frames, _ = _dfs_spec_geometry(dfs_bins)
    out = np.zeros((n_bins, n_frames), dtype=np.float32)
    amp = np.abs(csi).mean(axis=(2, 3))          # (T, 30)
    amp = amp - amp.mean(axis=0, keepdims=True)  # remove static/DC channel
    T = amp.shape[0]
    nper = min(DFS_NPERSEG, T)
    if nper < 4:
        return out.reshape(-1)
    _, _, Z = stft(amp, nperseg=nper, axis=0)    # (F, 30, frames)
    spec = np.abs(Z).mean(axis=1)                # (F, frames) subcarrier-avg
    k = min(n_bins, spec.shape[0])
    band = spec[:k, :]                            # (k, frames)
    n_fr = band.shape[1]
    if n_fr < 1:
        return out.reshape(-1)
    if n_fr == 1:
        # single frame — broadcast to fixed length
        out[:k, :] = band[:, :1]
        return out.reshape(-1)
    x_src = np.linspace(0.0, 1.0, n_fr)
    x_tgt = np.linspace(0.0, 1.0, n_frames)
    for b in range(k):
        out[b, :] = np.interp(x_tgt, x_src, band[b, :]).astype(np.float32)
    return out.reshape(-1)


def _receiver_feature(dat_path, feature="amp", dfs_bins="full"):
    """Return the per-receiver feature vector for one .dat, or None."""
    csi = _load_receiver_csi(dat_path)
    if csi is None:
        return None
    if feature == "dfs_spec":
        return _dfs_spec_block(csi, dfs_bins=dfs_bins).astype(np.float32)
    blocks = [_amp_block(csi)]
    if feature == "amp_phase":
        blocks.append(_phase_block(csi))
    elif feature == "amp_dfs":
        blocks.append(_dfs_block(csi))
    return np.concatenate(blocks).astype(np.float32)


def build_csi_features(csi_root, dates, keep_users=None, keep_gestures=None,
                       feature="amp", dfs_bins="full", verbose=True):
    """
    Walk the given date folders and assemble per-sample CSI features + labels.

    Args:
        csi_root:      base dir containing the date folders
        dates:         list of date-folder names, e.g. ["20181109","20181118"]
        keep_users:    optional set of user ids (ints) to keep
        keep_gestures: optional set of gesture ids (ints) to keep
        feature:       feature mode, one of FEATURE_MODES
                       ("amp" [default, 360], "amp_phase" [720],
                        "amp_dfs" [456], "dfs_spec" [1536, time preserved])
        dfs_bins:      dfs_spec size regime, one of DFS_BINS_MODES
                       ("full" [default, 16x16=256/rx -> 1536] or "small"
                        [5x5=25/rx -> 150] — a compact low-Doppler band
                        that gives a linear+magnitude readout a fair chance
                        to fit in-domain). Ignored for non-dfs_spec modes.

    Returns a dict of numpy arrays matching the b2 dump schema:
        X, y_room, y_location, y_orientation, y_user, y_gesture, groups
    """
    if feature not in FEATURE_MODES:
        raise ValueError(
            f"unknown feature mode {feature!r}; choose from {FEATURE_MODES}")
    if dfs_bins not in DFS_BINS_MODES:
        raise ValueError(
            f"unknown dfs_bins {dfs_bins!r}; choose from {DFS_BINS_MODES}")
    per_rx, feat_dim = feature_dim(feature, dfs_bins=dfs_bins)
    csi_root = Path(csi_root)
    samples = defaultdict(dict)   # key -> {rx_id: path}
    meta = {}                     # key -> (user,gesture,loc,ori,rep,room,date)

    for date in dates:
        date_dir = csi_root / date
        if not date_dir.exists():
            if verbose:
                print(f"[csi] MISSING date folder: {date_dir}")
            continue
        room = date_to_room(date)
        for root, _, files in os.walk(date_dir):
            for fn in files:
                if not fn.endswith(".dat"):
                    continue
                m = _FNAME_RE.match(fn[:-4])
                if not m:
                    continue
                uid, ges, loc, ori, rep, rx = (int(m.group(i)) for i in range(1, 7))
                if keep_users is not None and uid not in keep_users:
                    continue
                if keep_gestures is not None and ges not in keep_gestures:
                    continue
                if rx < 1 or rx > N_RX_FILES:
                    continue
                key = (date, uid, ges, loc, ori, rep)
                samples[key][rx] = os.path.join(root, fn)
                meta[key] = (uid, ges, loc, ori, rep, room, date)

    if not samples:
        raise RuntimeError("No CSI samples matched the given dates/filters.")

    X = []
    raw_user, raw_ges, raw_loc, raw_ori, raw_room, group_keys = [], [], [], [], [], []
    n_total = len(samples)
    n_done = 0
    for key in sorted(samples.keys()):
        rxmap = samples[key]
        vec = np.zeros(feat_dim, dtype=np.float32)
        got_any = False
        for rx in range(1, N_RX_FILES + 1):
            p = rxmap.get(rx)
            if p is None:
                continue
            fr = _receiver_feature(p, feature, dfs_bins=dfs_bins)
            if fr is None:
                continue
            vec[(rx - 1) * per_rx: rx * per_rx] = fr
            got_any = True
        if not got_any:
            continue
        uid, ges, loc, ori, rep, room, date = meta[key]
        X.append(vec)
        raw_user.append(uid)
        raw_ges.append(ges - 1)            # 0-based, matches BVP convention
        raw_loc.append(loc)
        raw_ori.append(ori)
        raw_room.append(room)
        group_keys.append(f"{uid}-{loc}-{ori}-{room}-{date}")
        n_done += 1
        if verbose and n_done % 200 == 0:
            print(f"[csi] {n_done}/{n_total} samples built")

    X = np.stack(X)

    def enc(vals):
        uniq = sorted(set(vals))
        mp = {v: i for i, v in enumerate(uniq)}
        return np.array([mp[v] for v in vals], dtype=np.int64), uniq

    y_room, room_set = enc(raw_room)
    y_location, loc_set = enc(raw_loc)
    y_orientation, ori_set = enc(raw_ori)
    y_user, user_set = enc(raw_user)
    y_gesture = np.array(raw_ges, dtype=np.int64)

    grp_set = sorted(set(group_keys))
    g2i = {g: i for i, g in enumerate(grp_set)}
    groups = np.array([g2i[g] for g in group_keys], dtype=np.int64)

    if verbose:
        extra = f" dfs_bins={dfs_bins}" if feature == "dfs_spec" else ""
        print(f"[csi] feature mode = {feature}{extra}  (per_rx={per_rx}, dim={feat_dim})")
        print(f"[csi] built {len(X)} samples, dim={X.shape[1]}")
        print(f"[csi] room classes {room_set} counts {dict(Counter(raw_room))}")
        print(f"[csi] user classes {user_set}")
        print(f"[csi] gesture counts {dict(Counter((y_gesture + 1).tolist()))}")
        print(f"[csi] location classes {loc_set}")
        print(f"[csi] orientation classes {ori_set}")
        print(f"[csi] groups: {len(grp_set)}")

    return {
        "X": X,
        "y_room": y_room,
        "y_location": y_location,
        "y_orientation": y_orientation,
        "y_user": y_user,
        "y_gesture": y_gesture,
        "groups": groups,
    }


def balance_by_room(data, seed=0, verbose=True):
    """Subsample the majority room(s) so every room has ~equal sample counts.

    Whole recordings (groups = user-loc-ori-room-date) are dropped as a unit,
    so no recording is split across the two rooms and per-group integrity is
    preserved. Because `room` is part of the group key, every group already
    belongs to exactly one room. All arrays whose first axis matches the number
    of samples are filtered; scalar/metadata entries are passed through.

    Returns a new dict with the balanced arrays. Prints the new per-room counts
    and the resulting chance level (majority-class fraction).
    """
    y_room = np.asarray(data["y_room"])
    groups = np.asarray(data["groups"])
    n = len(y_room)
    rooms, counts = np.unique(y_room, return_counts=True)
    if verbose:
        print(f"[balance] per-room counts before: {dict(zip(rooms.tolist(), counts.tolist()))}")
    if len(rooms) < 2:
        if verbose:
            print("[balance] fewer than 2 rooms present — nothing to balance.")
        return dict(data)

    target = int(counts.min())
    rng = np.random.default_rng(seed)
    keep = np.zeros(n, dtype=bool)

    for r in rooms:
        r_mask = y_room == r
        if int(r_mask.sum()) <= target:
            keep |= r_mask
            continue
        r_groups = np.unique(groups[r_mask])
        rng.shuffle(r_groups)
        total = 0
        for g in r_groups:
            g_mask = (groups == g) & r_mask
            gsize = int(g_mask.sum())
            if total + gsize <= target:
                keep |= g_mask
                total += gsize
        if total == 0:  # every group larger than target: keep the smallest
            smallest = min(r_groups, key=lambda g: int(((groups == g) & r_mask).sum()))
            keep |= (groups == smallest) & r_mask

    out = {}
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[k] = arr[keep]
        else:
            out[k] = v

    if verbose:
        new_rooms, new_counts = np.unique(out["y_room"], return_counts=True)
        counts_map = dict(zip(new_rooms.tolist(), new_counts.tolist()))
        chance = new_counts.max() / new_counts.sum()
        print(f"[balance] per-room counts after:  {counts_map}")
        print(f"[balance] kept {int(keep.sum())}/{n} samples")
        print(f"[balance] chance level (majority room fraction) = {chance * 100:.1f}%")
    return out
