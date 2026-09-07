"""Cross-domain label / class-index alignment checks.

The five-class Widar3.0 evaluation depends on the model-class-index -> canonical
gesture-id mapping being identical for the source domain (train) and the target
domain (test) of every cross-domain split. If a downstream pipeline ever rebuilds
that mapping per-domain (e.g. sorting unique labels within each room), the
resulting confusion matrix reads as random despite the model working correctly.

This module provides two functions the runner (and the diagnostic tools) call
before evaluating any cross-domain arm:

* :func:`assert_cross_domain_class_maps_agree` — raises
  :class:`ClassMapMismatch` if the source and target mappings disagree, or if a
  test-fold class index never appeared in the training fold. Fails LOUDLY.

* :func:`audit_saved_run_class_maps` — walks a completed run directory,
  reads every ``metrics/test.json`` and its sibling ``metrics/validation.json``
  and returns a list of mismatches without touching the results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CROSS_DOMAIN_KINDS = frozenset({
    "cross_room", "leave_one_room_out", "held_out_date",
})


class ClassMapMismatch(RuntimeError):
    """Source / target class-index mapping disagreement in a cross-domain eval."""


@dataclass(frozen=True)
class ClassMapReport:
    split_id: str
    suite: str
    arm: str
    seed: int
    ok: bool
    reason: str
    train_map: dict[int, int]
    test_map: dict[int, int]


def _as_int_key(m: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in (m or {}).items()}


def assert_cross_domain_class_maps_agree(
    split_kind: str,
    class_index_to_gesture_train: dict[int, int],
    class_index_to_gesture_test: dict[int, int] | None,
    y_train: Sequence[int],
    y_test: Sequence[int],
    *,
    split_id: str = "",
) -> None:
    """Raise :class:`ClassMapMismatch` if source/target mappings disagree.

    ``class_index_to_gesture_test`` may be ``None`` when the caller built a
    single feature bundle for both domains (the standard runner path); in
    that case the mapping is inherited by construction and we only verify
    the label sets seen by train and test are compatible.
    """
    if split_kind not in CROSS_DOMAIN_KINDS:
        return

    train_map = _as_int_key(class_index_to_gesture_train)
    if class_index_to_gesture_test is not None:
        test_map = _as_int_key(class_index_to_gesture_test)
        if train_map != test_map:
            raise ClassMapMismatch(
                f"split={split_id!r} kind={split_kind}: "
                f"class_index_to_gesture disagrees. "
                f"train={train_map} test={test_map}"
            )
    else:
        test_map = train_map

    seen_train = {int(v) for v in np.unique(np.asarray(y_train))}
    seen_test = {int(v) for v in np.unique(np.asarray(y_test))}
    unknown = sorted(seen_test - set(train_map))
    if unknown:
        raise ClassMapMismatch(
            f"split={split_id!r} kind={split_kind}: test-fold class indices "
            f"{unknown} are not in the training class map {sorted(train_map)}"
        )
    train_gestures = {train_map[c] for c in seen_train if c in train_map}
    test_gestures = {test_map[c] for c in seen_test if c in test_map}
    unmatched = sorted(test_gestures - train_gestures)
    if unmatched:
        raise ClassMapMismatch(
            f"split={split_id!r} kind={split_kind}: test fold contains "
            f"canonical gestures {unmatched} unseen in train fold. "
            f"train_gestures={sorted(train_gestures)} test_gestures={sorted(test_gestures)}"
        )


def _load_json(p: Path) -> dict | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def audit_saved_run_class_maps(run_root: Path) -> list[ClassMapReport]:
    """Walk a completed run and check every cross-domain experiment's map.

    Reads the split's ``kind`` from ``splits/<split_id>.json`` and the
    per-experiment saved ``class_index_to_gesture`` from
    ``metrics/test.json`` and ``metrics/validation.json``. In-domain
    experiments are skipped (they would trivially match).
    """
    run_root = Path(run_root)
    splits_dir = run_root / "splits"
    split_kinds: dict[str, str] = {}
    if splits_dir.exists():
        for p in splits_dir.glob("*.json"):
            d = _load_json(p) or {}
            split_kinds[str(d.get("split_id", p.stem))] = str(d.get("kind", ""))

    out: list[ClassMapReport] = []
    experiments = run_root / "experiments"
    if not experiments.exists():
        return out
    for status_path in experiments.glob("*/*/*/*/status.json"):
        exp_dir = status_path.parent
        suite = exp_dir.parent.parent.parent.name
        split_id = exp_dir.parent.parent.name
        arm = exp_dir.parent.name
        try:
            seed = int(exp_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        kind = split_kinds.get(split_id, "")
        if kind not in CROSS_DOMAIN_KINDS:
            continue
        test_json = _load_json(exp_dir / "metrics" / "test.json") or {}
        val_json = _load_json(exp_dir / "metrics" / "validation.json") or {}
        val_best = val_json.get("best") if isinstance(val_json.get("best"), dict) else {}
        train_map = _as_int_key(val_best.get("class_index_to_gesture") or {})
        test_map = _as_int_key(test_json.get("class_index_to_gesture") or {})
        if not train_map or not test_map:
            out.append(ClassMapReport(
                split_id=split_id, suite=suite, arm=arm, seed=seed,
                ok=False,
                reason="missing class_index_to_gesture on train or test artefact",
                train_map=train_map, test_map=test_map,
            ))
            continue
        ok = train_map == test_map
        out.append(ClassMapReport(
            split_id=split_id, suite=suite, arm=arm, seed=seed,
            ok=ok,
            reason="ok" if ok else f"train={train_map} != test={test_map}",
            train_map=train_map, test_map=test_map,
        ))
    return out


def summarise_reports(reports: Iterable[ClassMapReport]) -> dict:
    reports = list(reports)
    bad = [r for r in reports if not r.ok]
    return {
        "n_cross_domain_experiments": len(reports),
        "n_map_mismatches": len(bad),
        "mismatches": [
            {"suite": r.suite, "split_id": r.split_id, "arm": r.arm,
             "seed": r.seed, "reason": r.reason}
            for r in bad
        ],
    }
