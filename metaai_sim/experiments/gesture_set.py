"""Explicit gesture-set configuration.

A run must state exactly which canonical gesture ids are included and the
justification for that choice. The active set changes chance and majority
baselines, so both are recomputed per variant — never inherited.

Justification is persisted alongside the split definition and echoed into
the run manifest, resolved_config, and every summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from experiments.manifest import CANONICAL_GESTURES, RecordingRecord


# Canonical six-gesture pool the paper uses (matches ``gesture_map.py``
# entries for 20181109 and 20181118).
DEFAULT_SIX_CLASS = [1, 2, 3, 4, 5, 6]
FIVE_CLASS_NO_G4 = [1, 2, 3, 5, 6]


@dataclass(frozen=True)
class GestureSet:
    """Declared gesture ids + machine-readable justification.

    ``ids`` is the sorted set of canonical gesture ids (1-based) included
    in the run. ``justification`` records why — typed as a string keyed by
    convention (``"six-class-default"``, ``"five-class-drop-g4"``, ...).
    ``evidence`` points to the on-disk artefact that supports the choice.
    """

    ids: tuple[int, ...]
    label: str
    justification: str
    evidence: str = ""
    provenance: str = ""

    @classmethod
    def from_config(cls, cfg: dict | None) -> "GestureSet":
        cfg = dict(cfg or {})
        ids = tuple(sorted(int(g) for g in cfg.get("ids", DEFAULT_SIX_CLASS)))
        if not ids:
            raise ValueError("gesture_set.ids must not be empty")
        for g in ids:
            if g not in CANONICAL_GESTURES:
                raise ValueError(
                    f"gesture id {g} is not in CANONICAL_GESTURES "
                    f"{sorted(CANONICAL_GESTURES)}"
                )
        label = str(cfg.get("label") or f"{len(ids)}-class")
        justification = str(
            cfg.get("justification")
            or "six-class-default (Widar3.0 paper pool, gestures 1-6 "
               "identical across active dates per gesture_map.py)"
        )
        evidence = str(cfg.get("evidence") or "docs/gesture_set_evidence.md")
        provenance = str(cfg.get("provenance") or "")
        return cls(
            ids=ids, label=label, justification=justification,
            evidence=evidence, provenance=provenance,
        )

    @property
    def n_classes(self) -> int:
        return len(self.ids)

    @property
    def chance_level(self) -> float:
        """Uniform chance = 1/K; only correct baseline when the target-fold
        class distribution is uniform. The runner also reports the
        target-fold majority-class accuracy — never use one without the
        other."""
        return 1.0 / self.n_classes

    def to_dict(self) -> dict:
        return {
            "ids": list(self.ids),
            "label": self.label,
            "n_classes": self.n_classes,
            "chance_level": self.chance_level,
            "justification": self.justification,
            "evidence": self.evidence,
            "provenance": self.provenance,
            "gesture_names": {g: CANONICAL_GESTURES[g] for g in self.ids},
        }


def filter_records(
    records: Sequence[RecordingRecord],
    gset: GestureSet,
) -> list[RecordingRecord]:
    keep = set(gset.ids)
    return [r for r in records if r.gesture in keep]


def per_split_baseline(y_true: Iterable[int], gset: GestureSet) -> dict:
    """Chance + majority baselines under the active gesture set."""
    from experiments import metrics as M
    import numpy as np

    y = np.asarray(list(y_true), dtype=int)
    majority = M.majority_class_baseline(y)
    return {
        "gesture_set_label": gset.label,
        "n_classes": gset.n_classes,
        "chance_level": gset.chance_level,
        "majority_baseline_accuracy": majority["accuracy"],
        "majority_class": majority["majority_class"],
        "class_support": majority["class_support"],
        "n": majority["n"],
    }
