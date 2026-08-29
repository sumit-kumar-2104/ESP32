"""
Widar3.0 raw CSI loader for the B2 domain-predictability probe.

Reads Intel 5300 .dat files (one per receiver r1..r6), builds one fixed-size
feature vector per gesture sample, and attaches domain metadata:
    room        -> from the date folder (IEEE DataPort README room map)
    user/gesture/location/orientation/rep -> parsed from the filename
        user{id}-{gesture}-{loc}-{ori}-{rep}-r{rx}.dat

Feature per sample (subcarrier-resolved, antenna-averaged, 6 receivers):
    for each receiver r1..r6:
        amp = |CSI| averaged over antenna pairs   -> (T, 30)
        [ mean_t(amp) (30) , std_t(amp) (30) ]    -> 60
    concatenated in fixed order r1..r6 (missing receiver = zeros) -> 360

The subcarrier axis is kept because the room signature is per-frequency (the
core B1 argument). Time-mean captures the static channel (room); time-std
captures motion energy (gesture, used by the control probe).
"""

import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import csiread

N_SUB = 30
N_RX_FILES = 6                      # receivers r1..r6
FEAT_PER_RX = 2 * N_SUB            # mean + std over time
FEAT_DIM = N_RX_FILES * FEAT_PER_RX  # 360

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


def _receiver_feature(dat_path: str):
    """Return a 60-dim [mean_t, std_t] amplitude vector for one .dat, or None."""
    try:
        r = csiread.Intel(dat_path)
        r.read()
        csi = r.get_scaled_csi()          # (count, 30, nrx, ntx)
    except Exception:
        return None
    if csi is None or csi.ndim != 4 or csi.shape[0] == 0 or csi.shape[1] != N_SUB:
        return None
    amp = np.abs(csi).mean(axis=(2, 3))   # (count, 30)
    mean_t = amp.mean(axis=0)             # (30,)
    std_t = amp.std(axis=0)               # (30,)
    return np.concatenate([mean_t, std_t]).astype(np.float32)


def build_csi_features(csi_root, dates, keep_users=None, keep_gestures=None,
                       verbose=True):
    """
    Walk the given date folders and assemble per-sample CSI features + labels.

    Args:
        csi_root:      base dir containing the date folders
        dates:         list of date-folder names, e.g. ["20181109","20181118"]
        keep_users:    optional set of user ids (ints) to keep
        keep_gestures: optional set of gesture ids (ints) to keep

    Returns a dict of numpy arrays matching the b2 dump schema:
        X, y_room, y_location, y_orientation, y_user, y_gesture, groups
    """
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
        vec = np.zeros(FEAT_DIM, dtype=np.float32)
        got_any = False
        for rx in range(1, N_RX_FILES + 1):
            p = rxmap.get(rx)
            if p is None:
                continue
            fr = _receiver_feature(p)
            if fr is None:
                continue
            vec[(rx - 1) * FEAT_PER_RX: rx * FEAT_PER_RX] = fr
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
