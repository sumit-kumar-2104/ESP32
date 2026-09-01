# Raw Widar3.0 CSI — directory layout expected by this repo

This note documents the *raw* Widar3.0 CSI directory layout that the CSI
loader (`data/csi_loader.py`) and the `b2_dump_csi.py --input raw_csi`
opt-in expect. It applies to Phase 2 of `fix/indomain-and-c1`.

## Where to place the tree

Resolution order used by `config.get_raw_csi_dir()` (highest priority first):

1. `METAAI_RAW_CSI_DIR` environment variable.
2. `METAAI_DATA_DIR / widar3 / CSI` (where `METAAI_DATA_DIR` itself defaults
   to `~/.cache/metaai_data` if unset).

Example on the lab server (conda env `tf`):

```bash
export METAAI_DATA_DIR=~/scratch/metaai_data
# raw CSI must live at: ~/scratch/metaai_data/widar3/CSI/
```

If the directory does **not** exist or is empty when `--input raw_csi` is
requested, `config.require_raw_csi_dir()` prints a loud fatal message and
exits — no silent fallback to DFS or BVP. That is by design.

## Expected directory layout

```
<CSI root>/
├── 20181109/                      # date folder, YYYYMMDD prefix required
│   ├── user1-1-1-1-1-r1.dat       # one .dat file per receiver
│   ├── user1-1-1-1-1-r2.dat
│   ├── ...
│   └── user1-1-1-1-1-r6.dat
├── 20181117/
├── 20181118/
├── ...
└── 20181211/
```

Filename convention (matches `data/csi_loader.py::_FNAME_RE`):

```
user{U}-{gesture}-{loc}-{ori}-{rep}-r{rx}.dat
```

| field     | meaning                                                    |
|-----------|------------------------------------------------------------|
| `U`       | user id (positive integer)                                 |
| `gesture` | gesture id (1..6 for the paper set; date-dependent — see `gesture_map.py`) |
| `loc`     | location id                                                |
| `ori`     | orientation id                                             |
| `rep`     | repetition id                                              |
| `rx`      | receiver index (1..6)                                      |

Every recording is split across `--num-receivers = 6` `.dat` files. A
recording where any of `r1..r6` is missing is loaded as zero-padded for the
missing receiver only; a recording with no receivers at all is skipped.

## Where to download

- IEEE DataPort — Widar3.0 dataset: <https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset>
- Choose the **CSI (Intel 5300)** archive — the `.dat` files, not the
  pre-computed BVP `.mat` files.

## Sanity check after populating the tree

```bash
python b2_dump_csi.py --input raw_csi --feature dfs_spec --dfs-bins small \
    --dates 20181109 20181118 --balance-room --seed 42
```

Expected output near the top:

```
[csi-dump] root=/…/widar3/CSI   input-mode=raw_csi
[csi-dump] feature=dfs_spec dfs_bins=small balance_room=True
[csi-dump] feature dim = 150  (per_rx=25, receivers=6)
```

## Raw-CSI probe ceiling (Phase-0 addendum)

`data/csi_loader.py` also exposes `--feature raw`: subcarrier-resolved
amplitude (30 subcarriers) with the time axis linearly resampled to a fixed
`RAW_T_FRAMES = 32` frames per receiver, flattened. Six receivers ->
`30 * 32 * 6 = 5760` dim per sample. No STFT, no Doppler collapse, no
mean/std reduction — the closest to raw CSI we can hand a fixed-size
learner.

Use it via the Phase-0 diagnostic to measure the raw-CSI probe ceiling
directly (the target is ~80%+ from the earlier BVP pipeline):

```bash
python diagnose_indomain.py --features raw_csi --dates 20181109
```

The report saves to `logs/diag_raw_csi_<timestamp>.log` and compares the
sklearn LogReg + MLP probe accuracies against the DFS ~63% ceiling from
Phase 0.

## Gesture-id caveat (see `gesture_map.py`)

The numeric `gesture` field in the filename does **not** name the same
physical gesture on every date. `gesture_id=5` is `Draw-O(H)` on
`20181109` but `Draw-N(H)` on `20181117`. Any script that pools dates must
either restrict to the id range that is stable across the chosen dates
(e.g. 1..4) or apply a per-date remap. `diagnose_indomain.py` prints the
resolved table and fails loudly on an overload.
