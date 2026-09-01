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

## Honest state (2026-09-02, branch `fix/indomain-and-c1`)

- **DFS probe ceiling ~63% (MLP) / ~61% (LogReg)** in-domain. Realistic
  in-domain target on DFS features is therefore **~50-60%** for a
  linear + magnitude OTA model. The 80% number from the earlier BVP
  pipeline is **not** reachable on DFS features — that ceiling requires
  raw CSI (or BVP).
- **The recent server-run collapse was a TRAINING-LOOP / gradient bug**,
  not features, labels, normalization, or forward-path scale. Phase-0
  (`diagnose_indomain.py`) confirmed features + labels are fine and the
  OTA forward magnitudes are healthy. Task A (`diagnose_training.py`) is
  the next step — it instruments gradient norms + STE + softmax entropy
  to localize the dead path. Fix hooks (`fix_pipeline.enable_hook_STE`,
  `enable_hook_HEAD_LR`) are DISABLED by default until Task A points to
  one of them.
- **Raw CSI (Phase 2) is required to target ~80%**. Its probe ceiling is
  being verified in `diagnose_indomain.py --features raw_csi` before we
  invest in OTA retraining on it. Run that BEFORE enabling any raw-CSI
  gate at `--target-acc 0.70`.
- Downstream gated scripts (`c1_gate.py`, `b4_dose_response.py`,
  `leave_one_domain_out.py`) now default `--indomain-threshold` to 0.50
  (honest DFS floor) and accept `--target-acc` as an alias. For raw CSI,
  pass `--target-acc 0.70` (or higher) explicitly.

## Feature modes (`data/csi_loader.py`)

`--feature` selects the per-sample CSI feature vector (subcarrier axis kept):

| mode         | dim  | per receiver (×6)                                                                                                        |
|--------------|------|--------------------------------------------------------------------------------------------------------------------------|
| `amp` (def)  | 360  | `[mean_t(amp)(30), std_t(amp)(30)]` = 60                                                                                 |
| `amp_phase`  | 720  | amp(60) + sanitized-phase `[mean_t(30), std_t(30)]` = 60                                                                 |
| `amp_dfs`    | 456  | amp(60) + compact Doppler low-freq band = 16                                                                             |
| `dfs_spec`   | 1536 | Doppler-frequency spectrogram, `(DFS_SPEC_BINS=16, DFS_SPEC_FRAMES=16)` per rx = 256; **time axis preserved (no mean/std)** |

- `amp` is the original, unchanged behaviour.
- `amp_phase` unwraps phase across subcarriers and removes the per-packet linear
  slope (STO/CFO detrend) before taking time mean/std.
- `amp_dfs` takes an STFT along the packet/time axis of the DC-removed amplitude,
  averages magnitude over subcarriers and time frames, and keeps the first
  `DFS_BINS = 16` low-frequency bins.
- `dfs_spec` is the Doppler-**preserving** feature added for stage 5. For each
  receiver: DC-detrend the antenna-averaged amplitude, run an STFT along the
  packet/time axis per subcarrier, take magnitude, average over subcarriers to
  get a `(F, frames)` map, keep the first `DFS_SPEC_BINS=16` low-Doppler bins,
  and linearly resample the STFT time axis to a fixed `DFS_SPEC_FRAMES=16` (pad
  short recordings, downsample long ones). Output per receiver:
  `16 × 16 = 256`. Concatenated over 6 receivers → **1536 dim**. The temporal
  micro-Doppler axis is **not** collapsed to `mean_t/std_t`, which is what
  makes gesture/orientation content survive.

## Task 1 — Room-balanced dump

Subsamples the majority room to the minority room count (whole recordings only,
so no recording is split across rooms) and prints the new per-room counts and
the resulting chance level.

```bash
# amp features, room-balanced (chance ~50%)
python b2_dump_csi.py --feature amp --balance-room --seed 42

# phase-augmented, room-balanced
python b2_dump_csi.py --feature amp_phase --balance-room --seed 42

# Doppler-preserving spectrogram, room-balanced (the stage-5 feature)
python b2_dump_csi.py --balance-room --feature dfs_spec --seed 42
```

The dump records which `--feature` built it and whether it was balanced, and
prints the exact output dim (`per_rx=256, receivers=6, total=1536` for
`dfs_spec`) at dump time.

## Task 2 — Re-run the B2 domain probe on the fresh dump

`b2_probe.py` reads `dumps/csi.npz` and reports linear + MLP decodability of
room / location / orientation / user, plus a gesture control. Run it after
building each dump you want to compare:

```bash
python b2_probe.py --features csi
```

## Task 3 — Isolation experiment with all three models

Same balanced cross-room CSI dump and same grouped cross-room split; only the
computation changes:

- `OTA_linear`  — complex linear + `|.|` magnitude readout, argmax.
- `Digital_MLP` — small real MLP (Linear-ReLU-Linear-ReLU-Linear).
- `Digital_DANN` — same MLP backbone plus a room-domain head fed through a
  Gradient-Reversal Layer (Ganin & Lempitsky 2015). Cross-room training uses
  the source room labeled and the target room's features (labels unused) as
  the unlabeled target for the domain-invariance loss, scaled by
  `--lambda_dann` (default 0.5). In-domain CV uses only the training fold's
  own room labels as the invariance signal, so it never peeks at the test set.

Reports in-domain and cross-room accuracy + macro-F1 (mean±std over 3 seeds)
for every selected model, and probes room/location/user decodability from each
model's computed (penultimate/pre-argmax) features.

```bash
# All three models on the Doppler-preserving feature
python b5_isolation.py --features csi --balance-room --feature dfs_spec \
    --models OTA_linear Digital_MLP Digital_DANN --seed 42

# Any subset works; original two-model behaviour is the default
python b5_isolation.py --features csi --balance-room --feature dfs_spec \
    --models Digital_DANN --lambda_dann 0.8 --seed 42
```

Outputs (in `results/`):

- `b5_isolation_summary.csv` — appended row per (`feature_mode`, model) with
  `in_dom_acc, in_dom_f1, cross_room_acc, cross_room_f1,
  room_decodability, loc_decodability, user_decodability, room_chance,
  lambda_dann`. Different `--feature` runs never overwrite each other because
  the `feature_mode` column is part of every row.
- `b5_confusion_<model>_<feature>.png` / `.npy` — cross-room gesture confusion
  matrices per model, tagged with the feature mode.

If `dumps/csi.npz` already exists and was built with the same `--feature`, b5
reuses it; otherwise it rebuilds the CSI features from the raw tree. Pass
`--csi-root /path/to/widar3/CSI` to override the location.

## Existing B2 domain probe (unchanged defaults)

```bash
python b2_probe.py --features csi
```

## End-to-end recipe (`tf` env, stage-5 defaults)

```bash
conda activate tf
export METAAI_DATA_DIR=~/scratch/metaai_data
cd MPL/metaai_sim

# (a) build a dfs_spec balanced dump
python b2_dump_csi.py --balance-room --feature dfs_spec --seed 42

# (b) re-run b2_probe on the fresh dump
python b2_probe.py --features csi

# (c) run b5_isolation with all three models
python b5_isolation.py --features csi --balance-room --feature dfs_spec \
    --models OTA_linear Digital_MLP Digital_DANN --seed 42
```
