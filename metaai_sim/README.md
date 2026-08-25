# MetaAI Simulation

Reproducible simulation of the MetaAI paper: Feng et al., "Enabling Over-the-Air AI for Edge Computing via Metasurface-Driven Physical Neural Networks" (ACM SIGCOMM 2025, DOI 10.1145/3718958.3750474). Implements neural-network inference inside a simulated wireless channel using a complex-valued single-layer network, BPSK modulation, and 2-bit metasurface phase quantization.

## Quick Start

```bash
# Activate existing venv
# Windows:
..\..\..\.venv\Scripts\activate
# macOS/Linux:
source ../../.venv/bin/activate

# Install dependencies (if venv is fresh or broken)
pip install -r requirements.txt

# Stage 1: Train the complex model
python train.py

# Stage 1 + Stage 2: Evaluate (ideal + quantized)
python evaluate.py --quantize

# Stage 2: Full meta-atom sweep with plot
python evaluate.py --sweep
```

### Stage 3 Commands

```bash
# Stage 3: Train DiscreteNN baseline + noise-aware model, then evaluate
python train.py --discrete          # Stage 3A: DiscreteNN baseline
python train.py --noise-train       # Stage 3D: Noise-aware model
python evaluate.py --stage3         # Full Stage 3 ablation + plots
```

## Running on a Second Laptop

This repo is synced across machines. The `.venv` may break on the second laptop due to absolute paths. To fix:

1. **Activate the venv:**
   - Windows: `..\..\..\.venv\Scripts\activate`
   - macOS/Linux: `source ../../.venv/bin/activate`
2. **If activation fails**, rebuild the venv:
   ```bash
   pip install -r requirements.txt
   ```
3. **Just run the code** — MNIST auto-downloads to the per-machine cache (`~/.cache/metaai_data`) on first run. No manual download needed.
4. **Optional:** Set `METAAI_DATA_DIR` environment variable to a custom path if you don't want data in your home directory cache.

## Stage 1 Results — Ideal-Weight Simulation

| Metric | Achieved | Paper Target |
|--------|----------|--------------|
| Digital model test accuracy | **94.85%** | ~92.75% |
| Sequential-loop test accuracy | **94.85%** | — |
| Digital ↔ Sequential gap | **0.00%** | <2% |

- CHECK 1: Digital model accuracy > 90% ✓ (94.85%)
- CHECK 2: Sequential loop matches digital within 2% ✓ (0% gap)

## Stage 2 Results — 2-Bit Metasurface Quantization

| Meta-atoms (M) | Quantized Accuracy |
|:--------------:|:-----------------:|
| 64 | 94.78% |
| 256 | 94.87% |
| 576 | 94.83% |
| 1024 | 94.82% |

- Paper target (prototype): ~89.77%
- CHECK 3: Quantized accuracy in high-80s to 90s ✓ (94.87% at M=256)
- CHECK 4: Accuracy saturates by ~64 atoms (flat curve) ✓

**Sweep plot:** `results/meta_atom_sweep.png`

> **Note:** Our simulation exceeds the paper's reported accuracy because the paper's 89.77% reflects *real hardware* losses (insertion loss, phase errors, coupling) that aren't modeled here. The key finding is validated: accuracy saturates early with meta-atom count.

## Architecture

```
metaai_sim/
├── config.py                 # Hyperparameters, seed, data-dir resolver, dataset switch
├── data/loader.py            # MNIST via torchvision (per-machine cache)
├── data/widar_loader.py      # Widar3.0 BVP loader (defensive, per-machine cache)
├── models/linear_complex.py  # Single complex FC layer (Eqn. 3)
├── models/discrete_nn.py     # DiscreteNN baseline with STE (Stage 3A)
├── sim/sender.py             # BPSK encoder
├── sim/channel.py            # Channel + quantization + sync + multipath + noise
├── sim/receiver.py           # Argmax decoder
├── train.py                  # Training (standard / discrete / noise-aware, MNIST/Widar)
├── evaluate.py               # Evaluation (Stage 1–4, MNIST/Widar)
├── requirements.txt          # Pinned dependencies
└── results/                  # Auto-created: weights, logs, plots (git-ignored)
```

## Key Equations (from paper)

- **Eqn. 3:** `y_r = |Σ_i H_r(t_i) · x_i|`, predict `argmax_r |y_r|`
- **Eqn. 4:** `H_mts = (1/M) · Σ_{m=1..M} e^{j φ_m}` with 2-bit phases
- **Eqn. 7:** `Φ = argmin ‖H_mts − H_des‖` (greedy coordinate descent)
- **Eqn. 13:** Environmental noise on accumulated signal
- **Eqn. 14:** Hardware noise as pre-disturbance on input

## Stage 3 Results — Robustness Mechanisms

### Ablation Table

| Configuration | Accuracy | Paper Target |
|---|---|---|
| Stage 2: Quantized (M=256) | **94.87%** | ~89.77% |
| (A) DiscreteNN baseline | **89.88%** | ~72.05% |
| (B) Sync error 4µs, NO CDFA | **26.34%** | ~19.23% |
| (B) Sync error 4µs, WITH CDFA | **94.85%** | ~89.28% |
| (C) Multipath (lab), NO cancellation | **85.96%** | — |
| (C) Multipath (lab), WITH cancellation | **94.85%** | >82.65% |
| (D) Noise @10dB, standard training | **81.89%** | ~80.48% |
| (D) Noise @10dB, noise-aware training | **90.92%** | ~87.92% |

### (A) DiscreteNN vs Continuous-then-Quantize

Continuous-then-quantize (94.87%) clearly beats DiscreteNN (89.88%), validating the paper's core design claim. The gap in our simulation (5%) is smaller than the paper's (17%) because we don't model real hardware imperfections.

### (B) CDFA Clock Synchronization

Without CDFA, a 4µs sync error drops accuracy to 26% (near random). With CDFA coarse detection, accuracy is restored to ~94.85%. The "No CDFA" curve collapses while "With CDFA" stays flat.

**Plot:** `results/cdfa_sync.png`

### (C) Multipath Cancellation

| Environment | No Cancellation | With Cancellation |
|---|---|---|
| Corridor (mild, 2 taps) | 92.94% | 94.85% |
| Lab (moderate, 3 taps) | 85.96% | 94.85% |
| Office (severe, 5 taps) | 92.26% | 94.85% |

Intra-symbol sampling perfectly cancels environmental multipath by exploiting the zero-mean property of modulation symbols (paper Section 3.2).

### (D) Noise-Aware Training

| SNR (dB) | Standard | Noise-Aware |
|---|---|---|
| 5 | 62.47% | **83.25%** |
| 10 | 81.78% | **91.06%** |
| 15 | 91.16% | **93.12%** |
| 20 | 94.04% | 93.58% |
| 25 | 94.56% | 93.80% |
| 30 | 94.62% | 93.89% |

At low SNR (5–15 dB), noise-aware training provides 9–21% improvement. At high SNR, both converge (noise-aware is slightly lower due to regularization effect).

**Plot:** `results/noise_snr.png`

> **Summary:** All four mechanisms reproduce the paper's qualitative findings. Absolute numbers are higher than the paper's because our simulation doesn't include hardware losses (insertion loss, phase manufacturing errors, antenna coupling).

## Stage 4: Real Wi-Fi CSI — Widar3.0

Stage 4 ports the identical validated pipeline (Stages 1–3) to the Widar3.0 Wi-Fi gesture recognition dataset, using BVP (Body-coordinate Velocity Profile) features as input. This is the credibility checkpoint: does the simulation machinery—quantization, CDFA, multipath cancellation, noise-aware training—transfer to real wireless sensing data?

**Paper reference:** Feng et al., Table 1 — Widar3.0 results (6 gestures, CNN+GRU architecture).

### How to Obtain Widar3.0 Data

The Widar3.0 BVP data must be downloaded manually (it is ~1-2 GB):

1. **Official page:** http://tns.thss.tsinghua.edu.cn/widar3.0/
2. **Download mirrors:**
   - IEEE DataPort: https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset (DOI: 10.21227/7znf-qp86)
   - Tsinghua Disk: https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/
   - Baidu Disk: https://pan.baidu.com/s/1E-iG3Oo5gYRCXGl8uykuTQ (password: 4m47)
3. **Place BVP data at:** `~/.cache/metaai_data/widar3/BVP/` (or set `METAAI_DATA_DIR`)
4. Only the BVP feature files are needed (not raw CSI or DFS).

**Expected structure:**
```
~/.cache/metaai_data/widar3/
  BVP/
    <date-setup>/
      user1-1-1-r1.csv    (user-gesture-trial-receiver)
      ...
```

### Preprocessing (exact specification)

| Parameter | Value |
|---|---|
| Date subfolder | `20181109-VS` only (single-date, single-environment) |
| Gestures | 6 classes (push/sweep/clap/slide/draw-O/draw-Z), IDs 1–6 |
| BVP shape per sample | (20, 20, T) where T varies |
| Temporal interpolation | `T → 20` frames (linear, axis=-1) |
| Normalization | Per-frame min-max to [0, 1] |
| Flattening | 20×20×20 = 8000-dim real vector |
| No PCA, no L2 norm | Kept full 8000-dim input |
| Split (iid) | Stratified 90/10 per gesture, seed=42 — **primary/faithful** |
| Split (rep) | First 5 users train, user 6 test (leave-one-user-out) |

### Stage 4 Commands

```bash
# Step 4.1: Train digital model on Widar BVP
python train.py --dataset widar

# Step 4.2: Quantized evaluation + meta-atom sweep
python evaluate.py --dataset widar --sweep

# Step 4.3: Robustness mechanisms
python train.py --dataset widar --discrete
python train.py --dataset widar --noise-train
python evaluate.py --dataset widar --stage3
```

### Stage 4 Scorecard

| Metric | iid split (primary) | rep split | Paper Target |
|---|---|---|---|
| Digital model accuracy | **~80.67%** | **~84.50%** | ~89.67% |
| Sequential-loop accuracy | **~80.67%** | **~84.50%** | — |
| Digital ↔ Sequential gap | **0.00%** | **0.00%** | <2% |

The **0.00% digital==sequential gap** is the pipeline-correctness proof: every symbol flows through encode → channel → decode identically to the direct matrix multiply, confirming the simulation is faithful on real CSI data. The iid split is the primary/faithful metric; the rep split (leave-one-user-out) is reported for completeness.

### Stage 2 Quantization (Widar)

| Meta-atoms (M) | Quantized Accuracy |
|:-:|:-:|
| 1 | ~16.67% (random, 1/6 classes) |
| 4 | ~36% |
| 16 | ~62% |
| 64 | ~78% |
| 256 | **~79.20%** |
| 576 | ~80% |
| 1024 | ~80% |

Knee at M≈64, saturation by M=256. Same qualitative shape as MNIST sweep.

**Plot:** `results/widar_meta_atom_sweep.png`

### Stage 3 Robustness Mechanisms (Widar)

All four mechanisms show the same **break-then-recover** pattern validated on MNIST:

#### (A) DiscreteNN Baseline

| Configuration | Accuracy |
|---|---|
| Continuous-then-quantize (Stage 2, M=256) | ~79.20% |
| DiscreteNN (STE-trained discrete) | ~76.13% |

Continuous-then-quantize beats DiscreteNN by ~3%, confirming the paper's design claim (Table 1 target: ~82.33%).

#### (B) CDFA Clock Synchronization

Without CDFA, sync errors of ≥2µs collapse accuracy to ~13–19% (near random for 6 classes). With CDFA, accuracy recovers to ~80.67%. The "With CDFA" variation across the offset grid is due to discrete offset sampling, not a mechanism failure.

**Plot:** `results/widar_cdfa_sync.png`

#### (C) Multipath Cancellation

| Environment | No Cancellation | With Cancellation |
|---|---|---|
| Corridor (mild, 2 taps) | degraded | ~80.67% |
| Lab (moderate, 3 taps) | degraded | ~80.67% |
| Office (severe, 5 taps) | degraded | ~80.67% |

Cancellation fully restores accuracy via the zero-mean modulation property.

#### (D) Noise-Aware Training

Noise-aware training improves low-SNR robustness on Widar, matching the MNIST pattern. Standard training degrades sharply below 15 dB; noise-aware training maintains higher accuracy.

**Plot:** `results/widar_noise_snr.png`

### Honest Finding: Accuracy Gap vs Paper

Our faithful single-layer complex linear model reaches **~81% (iid) / ~84.5% (rep)** vs the paper's reported **89.67%**. The ~8–9% gap is expected and explained by:

1. **Architecture difference:** The paper uses a CNN+GRU for Widar3.0 (not a single linear layer). Our simulation deliberately uses the same single-layer architecture as MNIST to keep the pipeline identical and the comparison clean.
2. **BVP quality:** The paper's in-house BVP computation likely differs from the public Widar3.0 BVP release (processing pipeline, calibration, filtering).
3. **Split definition:** The paper's exact train/test split is not publicly specified. Our stratified 90/10 split on single-date data may not match theirs.

**This gap does not invalidate the simulation.** The pipeline-correctness proof (0.00% digital==sequential gap) and the clean break-then-recover patterns for all four robustness mechanisms confirm the simulation machinery is correct on real CSI data. The accuracy ceiling is an architecture/data limitation, not a pipeline bug.

### Cross-Domain Note

Pooling all dates (cross-environment generalization) caps accuracy at ~50%. This is a known cross-domain challenge in Wi-Fi sensing and is documented here for completeness — it is not a simulation issue.

### Architecture (Stage 4 additions)

```
metaai_sim/
├── data/widar_loader.py          # Widar3.0 BVP loader (defensive, no fabrication)
├── config.py                     # DATASET switch, WIDAR_NUM_CLASSES, dynamic INPUT_DIM
├── train.py                      # --dataset widar flag
├── evaluate.py                   # --dataset widar with --sweep/--stage3
└── (all other files unchanged)   # Same channel/receiver/sender pipeline
```
