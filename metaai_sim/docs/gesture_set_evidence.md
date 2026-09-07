# Gesture-set evidence and decision

**Question.** Does gesture id `4` denote the same physical gesture on the two
recording dates the packaged Widar3.0 profile walks (`20181109` and
`20181118`)?

## Evidence

The canonical per-date gesture-id → gesture-name table is maintained in
[metaai_sim/gesture_map.py](../gesture_map.py) and derived from the
Widar3.0 dataset's own `Data instructions.pdf` (Table "Gesture ID for
each date"). The relevant excerpts:

| date       | id=1     | id=2  | id=3 | **id=4** | id=5      | id=6         |
|------------|----------|-------|------|----------|-----------|--------------|
| 20181109   | Push&Pull| Sweep | Clap | **Slide**| Draw-O(H) | Draw-Zigzag(H) |
| 20181118   | Push&Pull| Sweep | Clap | **Slide**| Draw-N(H) | Draw-O(H)      |

Both dates map gesture id `4` to **Slide**. Ids `1`, `2`, `3`, `4` are
identical across the two dates. The overload lives on ids `5` and `6`
(`Draw-O(H)` vs `Draw-N(H)`; `Draw-Zigzag(H)` vs `Draw-O(H)`).

Source of the mapping: `metaai_sim/gesture_map.py::GESTURE_MAP_BY_DATE`.
Upstream provenance: Widar3.0 IEEE DataPort release, `Data
instructions.pdf`, Table "Gesture ID for each date". `gesture_map.py`
also lists `20181112` and `20181116` as `UNVERIFIED_DATES` — neither is
in the active profiles, so no unverified evidence is in play.

## Decision

Gesture id `4` denotes the same physical gesture (*Slide*) on `20181109`
and `20181118`. The six-class configuration
(`ids=[1, 2, 3, 4, 5, 6]`) is the primary run configuration.

The active gesture set for a run is declared explicitly in the profile
under `gesture_set:` and echoed into every run summary, together with
its `justification`, `evidence`, and derived `chance_level` and
`majority_baseline` per split. Chance and majority baselines are
recomputed per variant and never reused across variants.

## Alternative five-class variant

The task specification asked for a runnable five-class variant with
gesture `4` excluded, in case equivalence could not be confirmed. Since
equivalence *is* confirmed above we do not treat the five-class variant
as primary, but it remains available at
`experiments/profiles/core_five_class.yaml` for a defence-in-depth
comparison and is exercised by the smoke fixture path via
`experiments.gesture_set.FIVE_CLASS_NO_G4`.

Historical note: earlier project work excluded gesture `4` following an
unverified guideline. That exclusion is not consistent with the per-date
mapping above and is retained only as an optional variant.

## Cross-date warning: ids 5 and 6

Ids `5` and `6` are the ones that differ between `20181109` and
`20181118`:

- id 5: `Draw-O(H)` (20181109) vs `Draw-N(H)` (20181118)
- id 6: `Draw-Zigzag(H)` (20181109) vs `Draw-O(H)` (20181118)

Any cross-date evaluation therefore compares against a target fold whose
class 5 and class 6 are *different physical gestures* than those the
model was trained on. This is a **labelling shift** on top of the
distribution shift, and it is one plausible cause of the near-chance
cross-domain accuracy. It is not treated as a defect of the harness —
the packaged data is what it is — but it MUST be disclosed alongside
every cross-date claim.

The runner records this in each cross-date split's `notes` under
`cross_date_label_overloads` and repeats it in `RUN_SUMMARY.txt`.
