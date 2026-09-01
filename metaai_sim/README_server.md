# Server run guide (conda env `tf`)

All commands run from `MPL/metaai_sim` after activating the environment and
pulling the repo. The raw Widar3.0 CSI is expected under
`$METAAI_DATA_DIR/widar3/CSI` (default `~/scratch/metaai_data`).

```bash
conda activate tf
export METAAI_DATA_DIR=~/scratch/metaai_data
cd MPL/metaai_sim
```

Every script below accepts `--seed`, prints the compute device (CPU/GPU) at
startup, and writes a timestamped log to `logs/`. Nothing is fabricated: if the
raw CSI (or a matching dump) is missing, the script fails loudly with the exact
command to run.

## Feature modes (`data/csi_loader.py`)

`--feature` selects the per-sample CSI feature vector (subcarrier axis kept):

| mode        | dim | per receiver (×6)                                        |
|-------------|-----|----------------------------------------------------------|
| `amp` (def) | 360 | `[mean_t(amp)(30), std_t(amp)(30)]` = 60                 |
| `amp_phase` | 720 | amp(60) + sanitized-phase `[mean_t(30), std_t(30)]` = 60 |
| `amp_dfs`   | 456 | amp(60) + compact Doppler low-freq band = 16             |

- `amp` is the original, unchanged behaviour.
- `amp_phase` unwraps phase across subcarriers and removes the per-packet linear
  slope (STO/CFO detrend) before taking time mean/std.
- `amp_dfs` takes an STFT along the packet/time axis of the DC-removed amplitude,
  averages magnitude over subcarriers and time frames, and keeps the first
  `DFS_BINS = 16` low-frequency bins.

## Task 1 — Room-balanced dump

Subsamples the majority room to the minority room count (whole recordings only,
so no recording is split across rooms) and prints the new per-room counts and
the resulting chance level.

```bash
# amp features, room-balanced (chance ~50%)
python b2_dump_csi.py --feature amp --balance-room --seed 42

# phase-augmented, room-balanced
python b2_dump_csi.py --feature amp_phase --balance-room --seed 42
```

The dump records which `--feature` built it and whether it was balanced.

## Task 3 — Isolation experiment (the key result)

Same balanced cross-room CSI dump and same grouped cross-room split; only the
computation changes: `OTA_linear` (complex linear + |.| magnitude, argmax) vs
`Digital_MLP` (small real MLP). Reports in-domain vs cross-room accuracy +
macro-F1 (mean±std over 3 seeds) for both, then probes room/location/user
decodability from each model's COMPUTED (penultimate/pre-argmax) features.

```bash
python b5_isolation.py --features csi --balance-room --feature amp_phase --seed 42
```

Outputs written to `results/`:

- `b5_isolation_summary.csv` — model × {in-dom acc, cross-room acc,
  room-decodability-from-computed-features, loc/user decodability, room chance}
- `b5_confusion_OTA_linear.png`, `b5_confusion_Digital_MLP.png` (+ `.npy`)

If `dumps/csi.npz` already exists and was built with the same `--feature`, b5
reuses it; otherwise it rebuilds the CSI features from the raw tree. Pass
`--csi-root /path/to/widar3/CSI` to override the location.

## Existing B2 domain probe (unchanged defaults)

```bash
python b2_probe.py --features csi
```
