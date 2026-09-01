"""
Build dumps/csi.npz for the B2 probe from raw Widar3.0 CSI.

Default is a controlled cross-room pair with users and gestures held constant
so that ROOM is the only varying domain factor:
    Room 1: 20181109    Room 2: 20181118
    shared users 2,3    shared gestures 1,2,3,5,6 (gesture 4 differs by room)

Usage:
    python b2_dump_csi.py
    python b2_dump_csi.py --dates 20181109 20181118 20181117
    python b2_dump_csi.py --csi-root /path/to/widar3/CSI --users 2 3
    python b2_dump_csi.py --feature amp_phase --balance-room --seed 42

Opt-in `--input raw_csi` (Phase 2):
    Force the loader to use RAW Widar3.0 CSI resolved via METAAI_RAW_CSI_DIR
    (or METAAI_DATA_DIR/widar3/CSI). If the raw tree is missing, the script
    fails LOUDLY with an explicit fix message — no silent fallback. See
    README_raw_csi.md for the expected directory layout.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data.csi_loader import (
    build_csi_features,
    balance_by_room,
    FEATURE_MODES,
    DFS_BINS_MODES,
    feature_dim,
)
from config import setup_logging, print_device, set_seed


def default_csi_root() -> Path:
    from config import get_raw_csi_dir
    return get_raw_csi_dir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi-root", default=None)
    ap.add_argument("--input", choices=["auto", "raw_csi"], default="auto",
                    help="'auto' (default) = legacy resolve (env var or "
                         "get_data_dir/widar3/CSI). 'raw_csi' = require the "
                         "raw Widar3.0 CSI tree and fail loudly if missing; "
                         "no silent fallback to DFS or BVP.")
    ap.add_argument("--dates", nargs="+", default=["20181109", "20181118"])
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3],
                    help="user ids to keep (empty = all)")
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6],
                    help="gesture ids to keep (empty = all)")
    ap.add_argument("--feature", choices=FEATURE_MODES, default="amp",
                    help="CSI feature mode (default amp = original behaviour)")
    ap.add_argument("--dfs-bins", choices=DFS_BINS_MODES, default="full",
                    dest="dfs_bins",
                    help="dfs_spec size regime: full (default, 1536-dim) or "
                         "small (~150-dim compact low-Doppler band). Only "
                         "affects --feature dfs_spec; ignored otherwise.")
    ap.add_argument("--balance-room", action="store_true",
                    help="subsample the majority room to match the minority "
                         "room count (whole recordings only), so chance ~50%%")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_logging("b2_dump_csi")
    print_device()
    set_seed(args.seed)

    # Resolve the CSI root. `--input raw_csi` promotes the check from "warn +
    # scan for .dat files" to "fail loudly with a fix message" if the raw
    # tree is missing. `--csi-root` still wins if the user passes it.
    if args.csi_root:
        csi_root = Path(args.csi_root)
        if args.input == "raw_csi" and not (csi_root.exists() and any(csi_root.iterdir())):
            from config import require_raw_csi_dir
            # Print the same diagnostic and stop.
            require_raw_csi_dir()
    elif args.input == "raw_csi":
        from config import require_raw_csi_dir
        csi_root = require_raw_csi_dir()
    else:
        csi_root = default_csi_root()

    out = Path(args.out) if args.out else Path(__file__).parent / "dumps" / "csi.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[csi-dump] root={csi_root}   input-mode={args.input}")
    print(f"[csi-dump] dates={args.dates} users={args.users} gestures={args.gestures}")
    print(f"[csi-dump] feature={args.feature} dfs_bins={args.dfs_bins} "
          f"balance_room={args.balance_room}")
    per_rx, feat_dim = feature_dim(args.feature, dfs_bins=args.dfs_bins)
    print(f"[csi-dump] feature dim = {feat_dim}  (per_rx={per_rx}, receivers={6})")

    data = build_csi_features(
        csi_root, args.dates,
        keep_users=set(args.users) if args.users else None,
        keep_gestures=set(args.gestures) if args.gestures else None,
        feature=args.feature,
        dfs_bins=args.dfs_bins,
    )
    data = dict(data)
    if args.balance_room:
        data = balance_by_room(data, seed=args.seed)
    # Record which mode / dfs_bins / balancing built this dump so downstream
    # scripts can detect a mismatch and rebuild instead of silently reusing.
    data["feature_mode"] = np.array(args.feature)
    data["dfs_bins"] = np.array(args.dfs_bins)
    data["balanced"] = np.array(bool(args.balance_room))

    np.savez(out, **data)
    print(f"[csi-dump] saved {out}  X.shape={data['X'].shape}")


if __name__ == "__main__":
    main()
