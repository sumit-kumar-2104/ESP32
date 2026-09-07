"""Metric utilities.

We compute metrics from persisted prediction files so summaries can never
drift from the artefact. All arrays are 1-D numpy int/float.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def per_class_recall(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int] | None = None,
) -> dict[int, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if classes is None:
        classes = sorted(np.unique(y_true).tolist())
    out: dict[int, float] = {}
    for c in classes:
        mask = y_true == c
        n = int(mask.sum())
        if n == 0:
            out[int(c)] = float("nan")
        else:
            out[int(c)] = float((y_pred[mask] == c).mean())
    return out


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = list(per_class_recall(y_true, y_pred).values())
    finite = [r for r in recalls if not np.isnan(r)]
    return float(np.mean(finite)) if finite else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    f1s: list[float] = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else float("nan")


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int] | None = None,
) -> tuple[list[int], np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if classes is None:
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return list(classes), cm


def class_support(y_true: np.ndarray) -> dict[int, int]:
    y_true = np.asarray(y_true, dtype=int)
    uniq, counts = np.unique(y_true, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def summarise_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_index_to_gesture: dict[int, int] | None = None,
) -> dict:
    """Compute the canonical metric bundle from arrays.

    ``class_index_to_gesture`` records the model-class -> canonical-gesture-id
    mapping so summaries can never accidentally re-index.
    """
    classes, cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class_recall": {str(k): v for k, v in per_class_recall(y_true, y_pred).items()},
        "class_support": {str(k): v for k, v in class_support(y_true).items()},
        "classes": [int(c) for c in classes],
        "confusion_matrix": cm.tolist(),
        "class_index_to_gesture": (
            {str(k): int(v) for k, v in (class_index_to_gesture or {}).items()}
        ),
        "n_samples": int(len(y_true)),
    }


def write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_predictions_csv(
    path: Path,
    sample_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
) -> None:
    """Persist per-sample predictions. probs (N, C) is optional."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["sample_id", "y_true", "y_pred"]
        if probs is not None:
            probs = np.asarray(probs)
            if probs.ndim != 2 or probs.shape[0] != len(sample_ids):
                raise ValueError(
                    f"probs shape {probs.shape} inconsistent with N={len(sample_ids)}"
                )
            header += [f"p_{c}" for c in range(probs.shape[1])]
        w.writerow(header)
        for i, sid in enumerate(sample_ids):
            row = [sid, int(y_true[i]), int(y_pred[i])]
            if probs is not None:
                row += [float(x) for x in probs[i]]
            w.writerow(row)
