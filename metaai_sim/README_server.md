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

- **DFS: plain-linear control 62% ≈ probe ceiling 63%; OTA penalty ~20pp
  = continuous overfit + quantized grad-starvation (STE works, not a
  dead gradient).**
  - Continuous `ComplexLinear` OVERFITS on `dfs_full` — train 100 / val 41
    vs plain-linear control val 62. Fix hook **R1** (weight decay, input
    dropout, optional complex-dim bottleneck, early-stop on val) is
    opt-in via `train_ab_dfs.py --enable R1`.
  - 2-bit `DiscreteComplexLinear` UNDER-LEARNS — grad through the complex
    layer ≈ 0.08 vs control ≈ 1.13 (~14× smaller), softmax entropy stuck
    near uniform. Fix hook **R2** (grad-scaled STE with `--qgrad-scale`,
    optional separate `--lr-complex`, optional hardtanh-clipped STE
    surrogate) is opt-in via `train_ab_dfs.py --enable R2`.
  - **STE is confirmed working** (`diagnose_training.py` A.2 check +
    Task-A grad-norms). Do NOT touch STE as a fix — the earlier
    `enable_hook_STE` / `enable_hook_HEAD_LR` scaffolding is left
    disabled and unused.
- **Raw CSI probe ceiling ≈ 95%** (Phase-0 LogReg 92.67% / MLP 94.67%,
  chance 20%, dim=5760). Phase-2 in-domain OTA target on raw CSI is
  ≈ 80% and is trained end-to-end by `train_raw_csi.py --target-acc 0.70`
  (fails LOUDLY if the raw Widar3.0 CSI directory is missing — no silent
  DFS fallback).
- **All in-domain loops fit StandardScaler on the TRAIN fold only.**
  `train_ab_dfs.py` and `train_raw_csi.py` assert
  `sc.n_samples_seen_ == len(X_train)` at runtime; leaked stats stop
  the run.
- Downstream gated scripts (`c1_gate.py`, `b3_cka.py`,
  `b4_dose_response.py`, `leave_one_domain_out.py`) default
  `--indomain-threshold` to 0.50 (honest DFS floor) and accept
  `--target-acc` as an alias. For raw CSI, pass `--target-acc 0.70` (or
  higher) explicitly.

## DFS OTA fix hooks — A/B on `dfs_full` (`train_ab_dfs.py`)

Runs one short in-domain train per invocation on `dfs_full` and prints
train/val curves, weight-grad norms, softmax entropy, and `|y|` mean.
Three arms:

- `--enable none` — baseline `ComplexLinear`, `--wd` / `--patience` still
  honoured. Reproduces the Task-A control number.
- `--enable R1` — `--wd`, `--dropout`, optional `--complex-dim`
  bottleneck, early-stop on val (`--patience`). Fixes the continuous
  overfit (train 100 / val 41).
- `--enable R2` — grad-scaled STE on `DiscreteComplexLinear`
  (`--qgrad-scale`, default 8.0), optional `--lr-complex`, optional
  `--ste hardtanh` surrogate. Fixes the quantized ~14× grad shrink.

`--enable` swaps only the model / optimizer wiring; data, seed, scaler
(train-only), and val split are identical across arms.

```bash
# baseline
python train_ab_dfs.py --dates 20181109 --epochs 40 \
    | tee logs/dfs_ab_baseline_$(date +%Y%m%d_%H%M%S).log

# R1 — continuous overfit fix
python train_ab_dfs.py --dates 20181109 --epochs 40 \
    --enable R1 --wd 1e-3 --dropout 0.4 --complex-dim 512 --patience 6 \
    | tee logs/dfs_ab_R1_$(date +%Y%m%d_%H%M%S).log

# R2 — quantized under-learn fix (2-bit + grad-scaled STE)
python train_ab_dfs.py --dates 20181109 --epochs 40 \
    --enable R2 --qgrad-scale 8.0 --lr-complex 5e-3 --ste hardtanh \
    | tee logs/dfs_ab_R2_$(date +%Y%m%d_%H%M%S).log
```

## Raw-CSI Phase-2 in-domain trainer (`train_raw_csi.py`)

Only `--input raw_csi` is accepted; there is no silent DFS fallback. The
loader asserts on missing raw Widar3.0 CSI and prints the exact
IEEE-DataPort download instructions. StandardScaler is fit on the TRAIN
fold only, asserted at runtime. Target-acc default 0.70; gated by
`assert_indomain_ok`.

```bash
python train_raw_csi.py --input raw_csi --dates 20181109 \
    --epochs 40 --target-acc 0.70 \
    | tee logs/raw_csi_baseline_$(date +%Y%m%d_%H%M%S).log

# R1 or R2 hooks are available with the same flags as train_ab_dfs.py:
python train_raw_csi.py --input raw_csi --dates 20181109 \
    --epochs 40 --target-acc 0.70 \
    --enable R1 --wd 1e-3 --dropout 0.4 --complex-dim 1024 --patience 6 \
    | tee logs/raw_csi_R1_$(date +%Y%m%d_%H%M%S).log
```

If the raw CSI directory is missing the script fails with a message
listing the exact `METAAI_RAW_CSI_DIR` / `METAAI_DATA_DIR` variables and
the IEEE-DataPort URL to download the Widar3.0 Intel-5300 `.dat` files
from. See `README_raw_csi.md` for the expected directory layout.

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
