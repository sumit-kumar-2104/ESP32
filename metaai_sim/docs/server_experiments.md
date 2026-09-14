# Server experiments — MetaAI wireless-sensing evaluation

Operator guide for running the leakage-resistant, resumable evaluation
harness under `experiments/`. This document is the ground-truth contract:
if the code and this file disagree, prefer the code and file a fix.

## 1. Required environment

- Conda env: `tf` (already provisioned on the shared server).
- Python 3.10+ with the packages already pinned in `requirements.txt`.
- No new install steps introduced by this branch — the harness only adds
  Python modules under `experiments/` and reuses the existing training
  primitives (`models/linear_complex.py`, `models/discrete_nn.py`,
  `data/csi_loader.py`, `fix_pipeline.py`).
- **New optional requirement**: `PyYAML`. Already covered by the existing
  `requirements.txt` transitively (via matplotlib/other deps); the runner
  imports it lazily and fails with a clear message if missing.

### Dataset expectations

- Real runs (`--profile core` and `--profile full`) need the Widar3.0
  raw Intel-5300 CSI tree at either:
  - `$METAAI_RAW_CSI_DIR/<date>/user{u}-{g}-{loc}-{ori}-{rep}-r{rx}.dat`, or
  - `$METAAI_DATA_DIR/widar3/CSI/<date>/...` if the first env var is unset.
- If both are unset or the tree is empty, the runner refuses to guess a
  fallback dataset and exits with code 2. See `README_raw_csi.md` for the
  download instructions.
- Only the dates listed in the profile are walked. `core.yaml` and
  `full.yaml` currently walk `20181109` and `20181118`.

## 2. Commands

Run from `MPL/metaai_sim/`:

```bash
conda activate tf
export METAAI_DATA_DIR=~/scratch/metaai_data

# Smoke — synthetic fixture, ~seconds. Not scientific results.
bash scripts/run_all.sh --profile smoke --device auto

# Core — raw-amplitude in-domain + cross-date (required), cross-room (optional).
bash scripts/run_all.sh --profile core --device auto

# Full — core + DFS ablations + leave-one-room-out.
bash scripts/run_all.sh --profile full --device auto

# Reproduction of Widar3.0 CNN-GRU on structure-preserving tensors.
# Reads BVP .mat files from METAAI_BVP_DIR (or <data>/widar3/BVP);
# if missing, bvp_tensor is enrolled as `unavailable` with the exact
# reason and the run continues on csi_tensor + dfs_tensor.
bash scripts/run_all.sh --profile repro --device auto

# Composable OTA constraint ablation on CNN-GRU. Gated: any tensor
# representation whose reproduction has not completed refuses to run
# with a clear "reproduction gate not completed" reason.
bash scripts/run_all.sh --profile ota_ablation --device auto

# Resume a previous run by ID (must be a directory under results/).
bash scripts/run_all.sh --resume <run-id> --device auto
```

On Windows, use the equivalent Python entry point:

```powershell
python scripts/run_all.py --profile smoke --device cpu
```

### Flags

| flag                 | meaning                                                                   |
|----------------------|---------------------------------------------------------------------------|
| `--profile <name>`   | `smoke` / `core` / `full`, or a full path to a `.yaml` profile file.       |
| `--resume <run-id>`  | Continue an existing run in `results/<run-id>/`.                          |
| `--device auto\|cpu\|cuda` | Device selection for torch arms. `auto` picks CUDA if present.       |
| `--results-root DIR` | Override `results/` root. Useful for isolated soak tests.                 |
| `--force`            | Re-run experiments even when `status.json` says `completed`.              |
| `--run-id NAME`      | Override auto-generated run ID (must match `[A-Za-z0-9_.-]+`).            |

## 3. Expected outputs

Every run creates:

```
results/<run-id>/
  run_manifest.json           git commit, launch cmd, seeds, counts, env
  resolved_config.yaml        profile after resolution (source-of-truth)
  environment.txt             versions + only allow-listed env vars
  status.json                 overall run status
  logs/orchestration.log      per-timestamp orchestration events
  data_audit/
    manifest.jsonl            one row per raw-CSI recording
    rejections.jsonl          quality-policy actions
    summary.json              cache-id + quality counts
  splits/<split-id>.json      persisted train/val/test sample-IDs + checks
  experiments/<suite>/<split-id>/<model>/seed_<seed>/
    config.yaml               arm + split + seed
    status.json               pending|running|completed|failed|unavailable
    logs/console.log
    metrics/epochs.csv        train_loss/train_acc/val_acc per epoch
    metrics/validation.json   best + final val metrics
    metrics/test.json         locked-test metrics (from selected checkpoint)
    predictions/test_predictions.csv    sample_id, y_true, y_pred, probs
    checkpoints/best.pt, last.pt
  summaries/
    experiment_index.csv
    aggregate_metrics.csv     mean/std per (arm, split) across seeds
    findings.md               status counts + honesty notes
```

Key invariants (verified by tests):

- **No leakage across partitions.** Splits are group-aware; overlapping
  `group_id`, `sample_id`, or file-hash raises `InvalidSplitError`.
- **Scaler fit train-only.** The runner asserts
  `sc.n_samples_seen_ == len(X_train)` before transforming val / test.
- **Test set touched once.** Val is used for early stop and best-checkpoint
  selection; test metrics come from the selected checkpoint only.
- **Cache IDs disjoint.** `raw` and `dfs_spec` produce different
  `cache_id` values, so features never overwrite each other.
- **Status is atomic.** Every status.json is written via
  `os.replace()`; a corrupt file is treated as *not completed*.

## 4. Failure interpretation

| status         | meaning                                                                                 |
|----------------|-----------------------------------------------------------------------------------------|
| `pending`      | Enumerated but not yet started (should be transient during a run).                      |
| `running`      | In progress. If seen after a run, the process crashed and the file was left in place.   |
| `completed`    | Selected checkpoint evaluated on the locked test set; metrics on disk.                  |
| `failed`       | Raised an exception. `reason` field explains. Run exits non-zero if suite is required.  |
| `unavailable`  | Prereqs missing (dataset, gestures, environments). Always flagged in `findings.md`.     |

`bash scripts/run_all.sh` returns:

- `0` — all required-suite experiments completed.
- `1` — at least one required-suite experiment failed. Optional-suite
  failures do NOT propagate to a non-zero exit.
- `2` — data manifest could not be built (dataset missing / unreadable).

## 5. Workload estimate

Estimate is based only on configured experiment counts, not invented
timings. Multiply by wall-time of a single arm-seed-split combo on the
target GPU.

| profile | suites | arms × splits × seeds | notes                                       |
|---------|--------|-----------------------|---------------------------------------------|
| smoke   | 1      | 1 × 2 × 2 = 4         | Synthetic fixture; seconds on CPU.          |
| core    | 3      | up to 5 × 4 × 3 = 60  | Cross-room is optional; disable if needed.  |
| full    | 5      | up to 5 × 7 × 3 = 105 | DFS + LORO added.                           |

Domain-learning / adaptation experiments (§8 of the task specification)
are NOT wired into the shipped profiles: DANN/CORAL/IRM require multiple
labelled source environments and there is currently only one date-room
pair in the packaged profiles. The existing `b5_isolation.py` covers
DANN independently and can be invoked separately; the runner will mark
domain-learning experiments as `unavailable` with the explicit reason
"insufficient source environments" until multi-room training data is
enumerated in a profile.

## 6. Scientific limitations and interpretation rules

- **Chance level.** The six-class task in this branch has a chance level
  of `1/6 = 16.67%`, not 20%. The earlier 20% reference described a
  *five-class* variant (gesture 4 excluded); it is not the correct
  baseline for the current six-class configuration. See
  [`docs/gesture_set_evidence.md`](gesture_set_evidence.md) for the
  gesture-4 decision and its evidence — gesture 4 = *Slide* on both
  `20181109` and `20181118` per `metaai_sim/gesture_map.py`, so the
  six-class configuration is primary. A five-class variant with gesture
  4 excluded remains runnable at
  `experiments/profiles/core_five_class.yaml`.
- **Active gesture set is explicit.** Every profile declares
  `gesture_set:` with `ids`, `label`, `justification` and `evidence`.
  Chance and majority baselines are recomputed **per split, per
  variant** and never inherited across variants. The active set is
  echoed into `run_manifest.json`, `resolved_config.yaml`, every
  per-split baseline JSON under `summaries/per_split_baselines/`, and
  the top-of-run `RUN_SUMMARY.txt`.
- **Current findings (persisted).**
  - In-domain locked-test accuracy: ~83.8% digital MLP, ~75.1% OTA
    baseline, ~74.8% OTA R1, ~74.1% OTA R2, ~70.4% logreg, against a
    ~16.7% majority baseline.
  - Cross-room 1↔2 accuracy: **at chance (16.6-17.2%)**, with
    near-constant predictions (collapse indicator fires; see
    [`experiments/collapse.py`](../experiments/collapse.py)).
- **Room / date confound.** Room and recording date are fully
  confounded in the packaged Widar3.0 tree (`20181109` = room 1,
  `20181118` = room 2). No experiment in this branch can attribute
  cross-domain failure to room specifically. The runner enumerates a
  `same_room_different_date` split kind and emits an inventory into
  `run_manifest.json::inventory`; when the packaged data cannot
  materialise it, the arms are enrolled as `unavailable` with the exact
  reason. Do not describe any confounded result as a room effect.
- **Duplicate splits are deduplicated.** `cross_room_1_to_2` and
  `heldout_test_20181118` resolve to identical membership under the
  packaged tree. The runner keeps one canonical row and records the
  other as `alias_of` in per-experiment `status.json` and in the run
  manifest. Only *canonical* completed rows contribute to
  `unique_completed`; alias rows contribute only to `executed_rows`.
- **Confirmatory vs exploratory.** The primary comparison per
  cross-domain split kind is pre-registered in
  [`experiments/stats.py::PRIMARY_COMPARISONS`](../experiments/stats.py).
  All other rows are exploratory; the diagnostic report reports raw,
  BH-FDR-adjusted and Holm-Bonferroni-adjusted p-values plus a
  maximum-statistic null so post-hoc-selected results are compared
  honestly. Deterministic replicates (fixed-seed sklearn logreg) count
  as ONE measurement. Any result within ± 2 accuracy points of `1/K` is
  never reported as significant regardless of p-value.
- **Historical 92.80% comparison.** Three factors changed together:
  five classes → six, users `[2, 3]` → `[1, 2, 3]`, split policy
  non-grouped random → recording-grouped. The repository does not
  reconstruct the earlier configuration; the old and new numbers are
  NOT presented as a controlled comparison. See
  [`docs/results_interpretation.md`](results_interpretation.md) for
  the full statement.
- **Every historical number in the top-level `README.md`** was collected
  on a different branch and split policy. Fresh runs of this harness
  produce their own metrics, and only those appear under `results/`.
- **Statistical equivalence / non-inferiority claims are NOT
  auto-emitted.** Interpretation requires an explicit configured test.
- Physical / hardware validation is not part of this branch. The
  hardware checklist lives at `docs/hardware_validation_checklist.md`
  as a template; no synthetic "measured hardware" numbers exist.

## 7. Local self-check

Before starting a long GPU job:

```bash
cd MPL/metaai_sim
python -m pytest tests/ -q
bash scripts/run_all.sh --profile smoke --device cpu
```

Both commands are fast and touch every code path in the harness without
requiring the real dataset. They exercise: split leakage checks, atomic
status writes, resume-safety, feature bundle building on synthetic data,
scaler-train-only assertion, and summary generation.

## 8. Not yet implemented / out of scope for this branch

Domain-generalisation / adaptation (§8 in the task specification) is
**prepared** in this branch but **not yet executed**:

- Configuration lives in
  [`experiments/profiles/domain_adaptation.yaml`](../experiments/profiles/domain_adaptation.yaml).
- Arm registry lives in
  [`experiments/domain_adaptation.py`](../experiments/domain_adaptation.py).
- Gating rules refuse a multi-environment method when only one training
  environment exists (exact reason: *"insufficient source
  environments"*), refuse UDA arms without unlabelled target features,
  and refuse everything until `tasks_1_3_ready: true` is set.
- Even when gating passes, the training loop itself is not wired; every
  arm enrols as `unavailable` with reason "domain-adaptation training
  loop not enabled in this profile (task 8: prepare, do not yet run)".
- The core profile does NOT include this file — running DA/DG requires
  `--profile domain_adaptation` explicitly.

The remaining §6 controlled representation ablations, §7 bounded DFS
diagnostics beyond the existing R1/R2 hooks, and §9 phase / physical
validation are enumerable via new profile files without touching runner
code. Adding them cleanly requires:

1. A DFS-oriented arm registry entry per ablation (temporal shuffle,
   time-averaged, static-background removal). Each needs a feature
   builder that documents whether it uses training-fold-only information.
2. A same-room / different-date split that is materialisable — the
   inventory report in `run_manifest.json::inventory` and the "INVENTORY
   & SPLIT ALIASES" block in `RUN_SUMMARY.txt` state whether that is
   currently possible; on the packaged data it is not.
3. A hardware measurement-record schema under
   `docs/hardware_validation_checklist.md` before any physical numbers
   are written.

These are documented here so the omission is explicit and downstream
readers do not assume completeness.


## 9. Structure-preserving CNN-GRU: reproduction + OTA ablation

Motivation and rules recorded in this branch (do not change without a
matching update to the tests):

- Cross-domain collapse in the flat-feature harness cannot be attributed
  to signal absence — the Widar3.0 paper (Zheng et al., MobiSys 2019,
  Section 6.4 + Figure 19) runs the SAME split (train and test collected
  separately in Room 1 and Room 2) through a CNN-GRU model that keeps
  the axis structure, and reports:
  - **raw CSI 40.2% | DFS 77.8% | BVP 92.4%** (in-domain overall 92.7%).
- The reproduction gate rebuilds three tensor representations from the
  packaged raw CSI (18×30×T for CSI, 6×F×T for DFS) plus the provided
  BVP `.mat` files (Vx×Vy×t₀, each snapshot L1-normalised to sum to 1,
  temporal axis linearly resampled to `t0`).
- Every reference number in `summaries/reproduction_report.md` is the
  paper's published figure, cited inline. Achieved numbers are computed
  from `metrics/test.json` — never hand-typed. A representation whose
  reference was not approached says so; the report treats 80% of the
  reference as a soft PASS.
- The OTA physical-constraint stack (complex weighting, magnitude
  readout, 2-bit quantization with STE, and the R1/R2 variants) is
  implemented as `models/constraints.py::ConstrainedHead`, applied to
  the CNN-GRU head via `apply_constraints_to_cnn_gru`. Each constraint
  is independently toggleable, so a single ablation row is one
  `(representation × constraint × split × seed)` cell.
- `ota_ablation` is **gated**: `experiments/runner.py::_count_repro_completions`
  reads persisted `status.json` files under the run tree and refuses to
  execute any tensor representation whose `repro_<mode>` suite has zero
  completed `cnn_gru_<mode>` rows. The refusal reason is echoed into
  `status.json` for every enrolled row, so the gate is visible in
  `RUN_SUMMARY.txt` too.
- The BVP source data (Widar3.0 `.mat` files) is NOT generated by the
  raw-CSI pipeline. If missing, the runner enrols `bvp_tensor` arms as
  `unavailable` with `reason=<path> not present. NEVER substituting a
  CSI-derived approximation.`. Set `METAAI_BVP_DIR` to override the
  search path.

Reading the reproduction report:

- Each representation has one section. The first table row is
  `indomain_grouped` and is directly compared to the paper's number
  (`gap = achieved - reference`, in accuracy points). Rows for other
  split kinds are cross-domain — the paper does not publish a
  representation-specific cross-room number for these exact splits, so
  no gap column is computed.
- Cross-room 2→1 in the packaged tree trains on ~1,049-1,259 samples
  and is data-scarce; it is reported but is not the primary comparison.
- Chance and majority baselines under the active gesture set are shown
  at the bottom of each representation section, sourced from the same
  per-split JSON blobs the flat harness produces.

Reading the OTA ablation:

- Every constraint variant carries the label of the config it was
  built with (`constraint_label` in `status.json`), so the table can
  be pivoted per (representation, constraint) directly from
  `summaries/experiment_index.csv`.
- The research question the ablation is designed to answer is: does the
  physical constraint cost MORE on a domain-independent representation
  (BVP) than on a domain-dependent one (raw CSI)? The flat-feature
  evidence in this branch hints at a large swing (OTA penalty 8.74pp
  on raw amplitude vs 25.41pp on DFS). Only the persisted numbers
  qualify as the answer.


## 10. Cross-suite split aliasing (dedup)

`experiments/dedup.py::dedup_splits` now accepts a shared
`known_fingerprints` dictionary that is threaded across suites in a
single run. If two configured splits — e.g. `cross_room_1_to_2`,
`heldout_test_20181118`, and `loro_room_2` — resolve to the same
`(train, val, test)` membership across different suites, the runner
now marks the later occurrences as `alias_of` the earlier canonical id
and excludes them from `unique_completed`. Previously the detector
computed matching fingerprints inside each suite but did not act on
cross-suite collisions; `split_dedup.json` reported `n_aliases: 0`
even when the fingerprints were byte-identical. `tests/test_dedup.py`
carries a regression test for the fixed behaviour.
