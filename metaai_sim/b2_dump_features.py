"""
B2 — Domain-predictability linear probe: feature dumping.

For each model, run a forward pass over the full Widar3.0 BVP dataset and save
an .npz with arrays:
  X             [N, D]  — frozen feature vectors
  y_room        [N]     — room/environment label (filename field 4)
  y_location    [N]     — location label (filename field 2)
  y_orientation [N]     — orientation label (filename field 3)
  y_user        [N]     — user identity label (filename field 0)
  y_gesture     [N]     — gesture label (0..5)
  groups        [N]     — unique recording ID (for GroupKFold splitting)

Feature sets saved to dumps/:
  raw.npz   — flattened BVP input (reference)
  ota.npz   — |<w, x>| class-score vector from ComplexLinear
  mlp.npz   — penultimate-layer activations from MLP2Layer (Stage 5.2)
  wicbr.npz — penultimate-layer embedding from DACN (Wi-CBR)

High domain decoding from a feature set means the domain factor is still
present (not invariant) in those features — the model has not learned to
discard domain-specific information.

Requirements: numpy, scipy, scikit-learn, torch
Usage:        python b2_dump_features.py
"""

import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import scipy.io
import scipy.ndimage
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.widar_loader import (
    NUM_GESTURES, BVP_KEY, T_FIXED, BUGGY_STEMS,
    _get_widar_dir, _is_buggy, _normalize_per_frame, _interpolate_time,
)

DUMPS_DIR = Path(__file__).parent / "dumps"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Data loading with full domain metadata
# ---------------------------------------------------------------------------

def load_widar_with_metadata():
    """Load Widar3.0 BVP data preserving all domain metadata fields."""
    from config import WIDAR_DATE_SCOPE

    widar_dir = _get_widar_dir()
    bvp_root = widar_dir / "BVP"
    if not bvp_root.exists():
        print(f"ERROR: BVP data not found at {bvp_root}")
        sys.exit(1)

    from data.widar_loader import VALID_DATES, IN_DOMAIN_DATE
    dates_to_use = [IN_DOMAIN_DATE] if WIDAR_DATE_SCOPE == "single" else VALID_DATES

    features, labels = [], []
    users, locations, orientations, rooms, rec_ids = [], [], [], [], []
    n_skip = 0

    for date_folder in dates_to_use:
        date_path = bvp_root / date_folder
        if not date_path.exists():
            continue
        for mat_file in sorted(date_path.rglob("*.mat")):
            stem = mat_file.stem
            if _is_buggy(stem):
                n_skip += 1
                continue
            parts = stem.split("-")
            if len(parts) < 5:
                continue
            try:
                gesture_id = int(parts[1])
            except ValueError:
                continue
            if gesture_id < 1 or gesture_id > NUM_GESTURES:
                n_skip += 1
                continue
            try:
                mat_data = scipy.io.loadmat(str(mat_file))
                if BVP_KEY not in mat_data:
                    continue
                bvp = mat_data[BVP_KEY]
                if bvp.ndim != 3 or bvp.shape[:2] != (20, 20) or bvp.shape[2] == 0:
                    continue
            except Exception:
                continue

            bvp_norm = _normalize_per_frame(bvp)
            bvp_interp = _interpolate_time(bvp_norm, T_FIXED)
            features.append(bvp_interp.flatten().astype(np.float32))
            labels.append(gesture_id - 1)
            users.append(parts[0])
            locations.append(parts[2])
            orientations.append(parts[3])
            rooms.append(parts[4] if len(parts) > 4 else "0")
            # Unique recording: user-location-orientation-room-date
            rec_ids.append(f"{parts[0]}-{parts[2]}-{parts[3]}-{rooms[-1]}-{date_folder}")

    X = np.stack(features)
    y_gesture = np.array(labels, dtype=np.int64)

    def encode_labels(strings):
        uniq = sorted(set(strings))
        mapping = {v: i for i, v in enumerate(uniq)}
        return np.array([mapping[s] for s in strings], dtype=np.int64), uniq

    y_room, room_set = encode_labels(rooms)
    y_location, loc_set = encode_labels(locations)
    y_orientation, ori_set = encode_labels(orientations)
    y_user, user_set = encode_labels(users)

    rec_set = sorted(set(rec_ids))
    rec2id = {r: i for i, r in enumerate(rec_set)}
    groups = np.array([rec2id[r] for r in rec_ids], dtype=np.int64)

    print(f"[b2-data] Loaded {len(X)} samples, {X.shape[1]}-dim")
    print(f"[b2-data] Domain factors:")
    print(f"    room:        {len(room_set)} classes {room_set}")
    print(f"    location:    {len(loc_set)} classes {loc_set}")
    print(f"    orientation: {len(ori_set)} classes {ori_set}")
    print(f"    user:        {len(user_set)} classes {user_set}")
    print(f"[b2-data] Gesture classes: {sorted(Counter(y_gesture.tolist()).keys())}")
    print(f"[b2-data] Unique recording groups: {len(rec_set)}")
    print(f"[b2-data] Skipped: {n_skip}")

    return X, y_gesture, y_room, y_location, y_orientation, y_user, groups


def _save_npz(path, X, y_room, y_location, y_orientation, y_user, y_gesture, groups):
    np.savez(path, X=X, y_room=y_room, y_location=y_location,
             y_orientation=y_orientation, y_user=y_user,
             y_gesture=y_gesture, groups=groups)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def dump_raw(X, y_gesture, y_room, y_location, y_orientation, y_user, groups):
    """Raw flattened BVP — the reference ceiling for domain information."""
    path = DUMPS_DIR / "raw.npz"
    _save_npz(path, X, y_room, y_location, y_orientation, y_user, y_gesture, groups)
    print(f"[raw] saved {path}  X.shape={X.shape}")


def dump_ota(X, y_gesture, y_room, y_location, y_orientation, y_user, groups):
    """Over-the-air class-score vector |<w, x>| from trained ComplexLinear."""
    from models.linear_complex import ComplexLinear
    ckpt = RESULTS_DIR / "widar_stage1_weights.pt"
    if not ckpt.exists():
        warnings.warn(f"OTA checkpoint not found: {ckpt} — skipping")
        return

    input_dim = X.shape[1]
    num_classes = 6
    model = ComplexLinear(input_dim, num_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            batch = X_t[i:i+256]
            bipolar = 2.0 * batch - 1.0
            x_complex = torch.complex(bipolar, torch.zeros_like(bipolar))
            out = torch.abs(torch.matmul(x_complex, model.complex_weight))
            scores.append(out.numpy())

    X_ota = np.concatenate(scores, axis=0)
    path = DUMPS_DIR / "ota.npz"
    _save_npz(path, X_ota, y_room, y_location, y_orientation, y_user, y_gesture, groups)
    print(f"[ota] saved {path}  X.shape={X_ota.shape}")


def _train_mlp(input_dim, num_classes, X, y_gesture):
    """Train MLP2Layer on Widar gesture task and save checkpoint."""
    from benchmark.models import MLP2Layer
    from torch.utils.data import DataLoader, TensorDataset

    print("[mlp] No checkpoint found — training MLP2Layer (200 epochs)...")
    model = MLP2Layer(input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_gesture, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    model.train()
    for epoch in range(1, 201):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_t).argmax(1) == y_t).float().mean().item()
            print(f"    epoch {epoch}: acc={acc*100:.1f}%")
            model.train()

    ckpt_path = RESULTS_DIR / "widar_mlp_weights.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[mlp] Saved checkpoint: {ckpt_path}")
    return model


def dump_mlp(X, y_gesture, y_room, y_location, y_orientation, y_user, groups):
    """Penultimate-layer (256-dim ReLU) activations from Stage 5.2 MLP2Layer."""
    from benchmark.models import MLP2Layer

    candidates = [
        RESULTS_DIR / "widar_mlp_weights.pt",
        RESULTS_DIR / "benchmark_mlp_weights.pt",
    ]
    ckpt = None
    for c in candidates:
        if c.exists():
            ckpt = c
            break

    input_dim = X.shape[1]
    num_classes = 6

    if ckpt is not None:
        model = MLP2Layer(input_dim, num_classes)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    else:
        model = _train_mlp(input_dim, num_classes, X, y_gesture)

    model.eval()

    # Hook on net[1] (ReLU output after first Linear → 256-dim penultimate)
    activations = []

    def hook_fn(_module, _input, output):
        activations.append(output.detach().cpu().numpy())

    handle = model.net[1].register_forward_hook(hook_fn)

    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            model(X_t[i:i+256])

    handle.remove()
    X_mlp = np.concatenate(activations, axis=0)
    path = DUMPS_DIR / "mlp.npz"
    _save_npz(path, X_mlp, y_room, y_location, y_orientation, y_user, y_gesture, groups)
    print(f"[mlp] saved {path}  X.shape={X_mlp.shape}")


def dump_wicbr(X, y_gesture, y_room, y_location, y_orientation, y_user, groups):
    """Penultimate-layer (1024-dim) embedding from DACN (Wi-CBR).

    Wi-CBR uses dual image inputs (phase STIFMM + DFS), not raw BVP.
    We load the trained checkpoint, run a forward pass over the image dataset,
    and extract the 1024-dim embedding before the final FC layer.
    The samples are aligned to BVP via the filename convention.
    """
    import os
    from pathlib import Path as _Path

    CHECKPOINTS_DIR = _Path(__file__).parent / "checkpoints"
    WICBR_DIR = _Path(__file__).parent / "benchmark" / "external" / "wicbr"
    PHASE_DIR = WICBR_DIR / "WIDAR_STIFMM"
    DFS_DIR = WICBR_DIR / "WIDAR_STIFMM_DFS"

    candidates = [
        CHECKPOINTS_DIR / "wicbr.pt",
        RESULTS_DIR / "widar_wicbr_weights.pt",
        RESULTS_DIR / "wicbr_weights.pt",
    ]
    ckpt = None
    for c in candidates:
        if c.exists():
            ckpt = c
            break
    if ckpt is None:
        warnings.warn(
            f"Wi-CBR (DACN) checkpoint not found (searched {[str(c) for c in candidates]}) — "
            "skipping. Run train_wicbr.py first."
        )
        return

    if not PHASE_DIR.exists() or not DFS_DIR.exists():
        warnings.warn(
            f"Wi-CBR image data not found at {PHASE_DIR} / {DFS_DIR} — skipping."
        )
        return

    # Import DACN model from train_wicbr (avoids duplication)
    from train_wicbr import DACN
    import torchvision.transforms as T
    from PIL import Image

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    # Load checkpoint
    model = DACN(num_classes=6)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    # Build sample list matching BVP metadata order.
    # BVP filenames: user{U}-{gesture}-{loc}-{ori}-{room}-r{rep}
    # Image filenames: {uid}-{gesture}-{loc}-{ori}-{rep}.jpg
    # We iterate in same order as BVP loading and collect embeddings for
    # samples that have corresponding images.
    from config import WIDAR_DATE_SCOPE
    from data.widar_loader import VALID_DATES, IN_DOMAIN_DATE, _get_widar_dir, _is_buggy, NUM_GESTURES, BVP_KEY

    widar_dir = _get_widar_dir()
    bvp_root = widar_dir / "BVP"
    dates_to_use = [IN_DOMAIN_DATE] if WIDAR_DATE_SCOPE == "single" else VALID_DATES

    embeddings = []
    emb_y_gesture, emb_y_room, emb_y_location, emb_y_orientation, emb_y_user = [], [], [], [], []
    emb_groups = []
    n_miss = 0

    def encode_labels(strings):
        uniq = sorted(set(strings))
        mapping = {v: i for i, v in enumerate(uniq)}
        return np.array([mapping[s] for s in strings], dtype=np.int64)

    # Collect all valid BVP stems and their metadata in order
    sample_meta = []
    for date_folder in dates_to_use:
        date_path = bvp_root / date_folder
        if not date_path.exists():
            continue
        for mat_file in sorted(date_path.rglob("*.mat")):
            stem = mat_file.stem
            if _is_buggy(stem):
                continue
            parts = stem.split("-")
            if len(parts) < 5:
                continue
            try:
                gesture_id = int(parts[1])
            except ValueError:
                continue
            if gesture_id < 1 or gesture_id > NUM_GESTURES:
                continue
            # Extract user id number from "userX"
            user_str = parts[0]
            if not user_str.startswith("user"):
                continue
            uid = int(user_str[4:])
            loc = int(parts[2])
            ori = int(parts[3])
            room = parts[4] if len(parts) > 4 else "0"
            # repetition is parts[5] if exists (e.g. "r1" -> 1)
            rep_str = parts[5] if len(parts) > 5 else "r1"
            rep = int(rep_str[1:]) if rep_str.startswith("r") else 1
            sample_meta.append({
                "uid": uid, "gesture": gesture_id, "loc": loc, "ori": ori,
                "room": room, "rep": rep, "user_str": user_str,
                "loc_str": parts[2], "ori_str": parts[3],
                "date": date_folder,
            })

    # Extract embeddings for matching images
    batch_phase, batch_dfs, batch_idx = [], [], []
    BATCH_SIZE = 32

    def flush_batch():
        nonlocal batch_phase, batch_dfs, batch_idx
        if not batch_phase:
            return
        p_batch = torch.stack(batch_phase)
        d_batch = torch.stack(batch_dfs)
        with torch.no_grad():
            _, emb = model(p_batch, d_batch)
        for i, idx in enumerate(batch_idx):
            embeddings.append(emb[i].numpy())
            meta = sample_meta[idx]
            emb_y_gesture.append(meta["gesture"] - 1)
            emb_y_room.append(meta["room"])
            emb_y_location.append(meta["loc_str"])
            emb_y_orientation.append(meta["ori_str"])
            emb_y_user.append(meta["user_str"])
            emb_groups.append(f"{meta['user_str']}-{meta['loc_str']}-{meta['ori_str']}-{meta['room']}-{meta['date']}")
        batch_phase, batch_dfs, batch_idx = [], [], []

    for idx, meta in enumerate(sample_meta):
        fname = f"{meta['uid']}-{meta['gesture']}-{meta['loc']}-{meta['ori']}-{meta['rep']}.jpg"
        p_path = PHASE_DIR / fname
        d_path = DFS_DIR / fname
        if not p_path.exists() or not d_path.exists():
            n_miss += 1
            continue
        p_img = transform(Image.open(str(p_path)).convert('RGB'))
        d_img = transform(Image.open(str(d_path)).convert('RGB'))
        batch_phase.append(p_img)
        batch_dfs.append(d_img)
        batch_idx.append(idx)
        if len(batch_phase) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    if len(embeddings) == 0:
        warnings.warn("No Wi-CBR embeddings extracted (no matching images found) — skipping.")
        return

    X_wicbr = np.stack(embeddings)
    y_g = np.array(emb_y_gesture, dtype=np.int64)

    # Encode string labels
    all_rooms = sorted(set(emb_y_room))
    room_map = {v: i for i, v in enumerate(all_rooms)}
    y_r = np.array([room_map[r] for r in emb_y_room], dtype=np.int64)

    all_locs = sorted(set(emb_y_location))
    loc_map = {v: i for i, v in enumerate(all_locs)}
    y_l = np.array([loc_map[l] for l in emb_y_location], dtype=np.int64)

    all_oris = sorted(set(emb_y_orientation))
    ori_map = {v: i for i, v in enumerate(all_oris)}
    y_o = np.array([ori_map[o] for o in emb_y_orientation], dtype=np.int64)

    all_users = sorted(set(emb_y_user))
    user_map = {v: i for i, v in enumerate(all_users)}
    y_u = np.array([user_map[u] for u in emb_y_user], dtype=np.int64)

    rec_set = sorted(set(emb_groups))
    rec_map = {r: i for i, r in enumerate(rec_set)}
    g_arr = np.array([rec_map[r] for r in emb_groups], dtype=np.int64)

    path = DUMPS_DIR / "wicbr.npz"
    _save_npz(path, X_wicbr, y_r, y_l, y_o, y_u, y_g, g_arr)
    print(f"[wicbr] saved {path}  X.shape={X_wicbr.shape}  "
          f"({n_miss} BVP samples had no matching image)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("B2 — Domain-predictability probe: dumping frozen features")
    print("=" * 60)

    X, y_gesture, y_room, y_location, y_orientation, y_user, groups = (
        load_widar_with_metadata()
    )
    DUMPS_DIR.mkdir(exist_ok=True)

    dump_raw(X, y_gesture, y_room, y_location, y_orientation, y_user, groups)
    dump_ota(X, y_gesture, y_room, y_location, y_orientation, y_user, groups)
    dump_mlp(X, y_gesture, y_room, y_location, y_orientation, y_user, groups)
    dump_wicbr(X, y_gesture, y_room, y_location, y_orientation, y_user, groups)

    print("\n[done] Feature dumps complete. Run b2_probe.py next.")


if __name__ == "__main__":
    main()
