# Results interpretation — cross-room / cross-date evaluation

This note states the defensible current claim in one place. Any external
communication that departs from these bounds is inconsistent with the
saved artefacts.

## Defensible current claim

Under the packaged Widar3.0 tree (dates `20181109` in room 1 and
`20181118` in room 2, six-class gesture pool), the harness under
`experiments/` produces the following persisted numbers:

- **In-domain (locked test), 20181109**
  - digital MLP (raw): ~83.8% (+/- ~0.5%)
  - OTA baseline (raw): ~75.1% (+/- ~2.6%)
  - OTA R1 (raw): ~74.8%
  - OTA R2 (raw): ~74.1%
  - Logistic regression (raw): ~70.4%
  - Six-class chance level: 1/6 = **16.67%**
  - Target-fold majority baseline: ~16.7%

- **Cross-room 1 → 2 and 2 → 1 (also cross-date by construction)**
  - Every arm collapses to near-constant predictions.
  - Observed accuracy 16.6-17.2%, i.e. at the six-class chance level and
    within the tolerance of the target-fold majority baseline.
  - For example: `cross_room 2 → 1 / logreg_raw` predicted counts
    `{0: 7496, 1: 1, 4: 1}`, with per-class recall of 0.00 on five of
    six classes. The collapse indicator (documented in
    `experiments/collapse.py`) fires at the default 0.95 top-share
    threshold.

**Claim.** In-domain accuracy is roughly 75-84% and falls to chance
across rooms, with near-constant predictions, for linear, digital neural
and constrained OTA models alike.

**Not claimed.**
- Robustness to domain shift.
- Any cross-room generalisation capability.
- A demonstrated causal mechanism for the collapse. Room and session are
  fully confounded in the packaged data (see below); no experiment in
  this branch can attribute the collapse to a room effect specifically.

## Confounds

- **Room and date are fully confounded** in the packaged tree. `20181109`
  is only in room 1 and `20181118` is only in room 2. A same-room /
  different-date evaluation is unavailable with the packaged data and is
  reported as such in the run manifest and RUN_SUMMARY. The harness will
  materialise the split as soon as an extra date is dropped into the raw
  CSI tree — no code change is needed.

- **Cross-date label overload on ids 5 and 6.** On `20181109` id 5 is
  `Draw-O(H)` and id 6 is `Draw-Zigzag(H)`; on `20181118` id 5 is
  `Draw-N(H)` and id 6 is `Draw-O(H)`. Any cross-date evaluation therefore
  compares against a target fold whose class 5 and class 6 are different
  physical gestures than those the model was trained on. This is a
  labelling shift on top of the distribution shift; the runner records it
  in the split notes and it is one plausible contributor to the collapse.
  Documented in [gesture_set_evidence.md](gesture_set_evidence.md).

- **Duplicate split alias.** `cross_room_1_to_2` and
  `heldout_test_20181118` produce identical `(train, val, test)` membership
  under the packaged data. The runner keeps one canonical row and records
  the other as `alias_of`; the aliased split is never counted as
  independent evidence.

## Confirmatory vs exploratory

- **Confirmatory (pre-registered in [`experiments/stats.py::PRIMARY_COMPARISONS`](../experiments/stats.py)):**
  the primary transfer arm on each cross-domain split is `digital_mlp_raw`,
  seed 42. Its shuffle-null p-value is reported unadjusted for the primary
  test only.
- **Exploratory:** every other row is labelled as such in the diagnostic
  report, and reports come with BH-FDR and Holm-Bonferroni corrected
  p-values alongside the raw ones. A maximum-statistic null is also
  reported so any post-hoc "best result" can be compared without lying
  about selection.
- **Guardrail:** any observed accuracy within ± 2 accuracy points of `1/K`
  is never reported as a significant finding regardless of p-value; see
  [`experiments/stats.py::format_finding`](../experiments/stats.py).

## Historical comparison

The often-cited 92.80% figure was measured on a different configuration.
Three factors changed together between that setting and the current
runs — **class count** (5 vs 6), **user set** (`[2, 3]` vs `[1, 2, 3]`),
and **split policy** (non-grouped random vs recording-grouped) — so the
drop cannot be attributed to any single factor. The repository does not
yet reconstruct the earlier configuration; the two numbers are NOT
presented as a controlled comparison. A `historical_replication`
profile that varies one factor at a time can be added later; when it
lands the reconstruction result will be recorded here.

## Prediction-collapse indicator (documented threshold)

`experiments/collapse.py::collapse_indicator` reports a boolean
`collapsed=True` when either

- the largest predicted-class share is ≥ 0.95, or
- fewer than 2 distinct classes are predicted on the target fold.

These thresholds are configurable via
`profile.collapse_top_share_threshold` and
`profile.collapse_min_classes_predicted`, and the actual values used are
echoed into every result. The `2 → 1 / logreg_raw` example above scores
`top_share ≈ 7496/7498 ≈ 0.999` and predicts only 3 distinct classes;
the indicator fires.

## Domain adaptation / generalisation

Not enabled in the core profile. A prepared but gated profile lives at
[`experiments/profiles/domain_adaptation.yaml`](../experiments/profiles/domain_adaptation.yaml).
It refuses to run multi-environment methods when only one training
environment exists ("insufficient source environments") and marks all
its arms as `unavailable` until the training loop is wired in
(task 8: prepare, do not yet run).
