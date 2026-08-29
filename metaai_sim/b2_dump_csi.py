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
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data.csi_loader import build_csi_features


def default_csi_root() -> Path:
    from config import get_data_dir
    return get_data_dir() / "widar3" / "CSI"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi-root", default=None)
    ap.add_argument("--dates", nargs="+", default=["20181109", "20181118"])
    ap.add_argument("--users", nargs="+", type=int, default=[2, 3],
                    help="user ids to keep (empty = all)")
    ap.add_argument("--gestures", nargs="+", type=int, default=[1, 2, 3, 5, 6],
                    help="gesture ids to keep (empty = all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csi_root = Path(args.csi_root) if args.csi_root else default_csi_root()
    out = Path(args.out) if args.out else Path(__file__).parent / "dumps" / "csi.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[csi-dump] root={csi_root}")
    print(f"[csi-dump] dates={args.dates} users={args.users} gestures={args.gestures}")

    data = build_csi_features(
        csi_root, args.dates,
        keep_users=set(args.users) if args.users else None,
        keep_gestures=set(args.gestures) if args.gestures else None,
    )
    np.savez(out, **data)
    print(f"[csi-dump] saved {out}  X.shape={data['X'].shape}")


if __name__ == "__main__":
    main()
