"""Prediction-collapse characterisation.

Computed only from persisted ``predictions/test_predictions.csv`` files —
never from a rerun. For every cross-domain experiment we report:

- predicted-class distribution + its (Shannon) entropy;
- largest predicted-class share;
- per-class recall and precision;
- a boolean ``collapsed`` indicator with a documented threshold;
- comparison against the target-fold majority baseline;
- source-domain performance of the same checkpoint (best_val_accuracy)
  for contrast.

Feature-shift diagnostics use the training fold only for any fitted
component; see :func:`compute_feature_shift_stats`.

Collapse is reported as an observed behaviour. Causes are ranked
elsewhere as hypotheses; never asserted as a single cause without a
controlled test.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Documented default. A "collapsed" model concentrates >=95% of its
# predictions in a single class OR predicts at most one class with
# recall >=1e-6 across all classes on the target fold. Tunable via the
# runner config, echoed into the summary.
DEFAULT_COLLAPSE_TOP_SHARE = 0.95
DEFAULT_COLLAPSE_MIN_CLASSES_PREDICTED = 2


@dataclass(frozen=True)
class CollapseIndicator:
    top_class_share: float
    n_classes_predicted: int
    entropy_bits: float
    max_entropy_bits: float
    top_share_threshold: float
    min_classes_predicted_threshold: int
    collapsed: bool


def prediction_distribution(y_pred: Iterable[int]) -> dict[int, int]:
    """Predicted-class counts, keyed by int class index."""
    return {int(k): int(v) for k, v in Counter(int(x) for x in y_pred).items()}


def shannon_entropy_bits(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        ent -= p * math.log2(p)
    return float(ent)


def per_class_precision(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int] | None = None,
) -> dict[int, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if classes is None:
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    out: dict[int, float] = {}
    for c in classes:
        mask = y_pred == c
        n = int(mask.sum())
        if n == 0:
            out[int(c)] = float("nan")
        else:
            out[int(c)] = float((y_true[mask] == c).mean())
    return out


def collapse_indicator(
    y_pred: Iterable[int],
    n_classes: int,
    *,
    top_share_threshold: float = DEFAULT_COLLAPSE_TOP_SHARE,
    min_classes_predicted: int = DEFAULT_COLLAPSE_MIN_CLASSES_PREDICTED,
) -> CollapseIndicator:
    """Boolean collapse decision + the numeric evidence.

    ``collapsed`` is True when *either*:

    - the largest predicted-class share is >= ``top_share_threshold``, or
    - fewer than ``min_classes_predicted`` classes are predicted at all.

    The threshold defaults were chosen so a well-behaved 5- or 6-class
    classifier on a roughly balanced target fold cannot trip them by
    accident. The runner records the actual values used in each result.
    """
    counts = prediction_distribution(y_pred)
    total = sum(counts.values())
    if total == 0:
        return CollapseIndicator(
            top_class_share=0.0, n_classes_predicted=0,
            entropy_bits=0.0,
            max_entropy_bits=math.log2(n_classes) if n_classes > 1 else 0.0,
            top_share_threshold=top_share_threshold,
            min_classes_predicted_threshold=min_classes_predicted,
            collapsed=True,
        )
    top_share = max(counts.values()) / total
    n_predicted = len(counts)
    collapsed = (
        top_share >= top_share_threshold
        or n_predicted < min_classes_predicted
    )
    return CollapseIndicator(
        top_class_share=float(top_share),
        n_classes_predicted=int(n_predicted),
        entropy_bits=shannon_entropy_bits(counts),
        max_entropy_bits=math.log2(n_classes) if n_classes > 1 else 0.0,
        top_share_threshold=float(top_share_threshold),
        min_classes_predicted_threshold=int(min_classes_predicted),
        collapsed=bool(collapsed),
    )


@dataclass
class CollapseReport:
    suite: str
    split_id: str
    split_kind: str
    direction: str
    arm: str
    seed: int
    n_test: int
    n_classes_in_task: int
    predicted_counts: dict[int, int]
    predicted_share: dict[int, float]
    largest_predicted_class: int | None
    largest_predicted_share: float
    entropy_bits: float
    max_entropy_bits: float
    per_class_recall: dict[int, float]
    per_class_precision: dict[int, float]
    observed_accuracy: float
    majority_baseline_accuracy: float
    majority_class: int | None
    accuracy_minus_majority: float
    chance_level: float
    source_domain_best_val_accuracy: float | None
    source_domain_final_val_accuracy: float | None
    collapse_indicator: dict = field(default_factory=dict)


def _load_predictions_csv(p: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    y_t: list[int] = []
    y_p: list[int] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                y_t.append(int(row["y_true"]))
                y_p.append(int(row["y_pred"]))
            except (KeyError, ValueError):
                continue
    if not y_t:
        return None
    return np.asarray(y_t, dtype=int), np.asarray(y_p, dtype=int)


def _load_json(p: Path) -> dict | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def analyse_experiment(
    exp_dir: Path,
    *,
    suite: str,
    split_id: str,
    split_kind: str,
    direction: str,
    arm: str,
    seed: int,
    n_classes_in_task: int,
    top_share_threshold: float = DEFAULT_COLLAPSE_TOP_SHARE,
    min_classes_predicted: int = DEFAULT_COLLAPSE_MIN_CLASSES_PREDICTED,
) -> CollapseReport | None:
    """Analyse a single completed experiment's predictions on disk.

    Returns ``None`` when the predictions file is missing/empty. Never
    reruns training or rewrites results.
    """
    from experiments import metrics as M

    pair = _load_predictions_csv(exp_dir / "predictions" / "test_predictions.csv")
    if pair is None:
        return None
    y_true, y_pred = pair
    n = int(y_true.size)
    counts = prediction_distribution(y_pred)
    total = sum(counts.values())
    share = {int(k): float(v) / max(total, 1) for k, v in counts.items()}
    top_class = max(counts, key=counts.get) if counts else None
    top_share = max(share.values()) if share else 0.0
    entropy = shannon_entropy_bits(counts)
    max_ent = math.log2(n_classes_in_task) if n_classes_in_task > 1 else 0.0
    recalls = M.per_class_recall(y_true, y_pred, classes=range(n_classes_in_task))
    precs = per_class_precision(y_true, y_pred, classes=range(n_classes_in_task))
    baseline = M.majority_class_baseline(y_true)
    ci = collapse_indicator(
        y_pred, n_classes=n_classes_in_task,
        top_share_threshold=top_share_threshold,
        min_classes_predicted=min_classes_predicted,
    )
    val = _load_json(exp_dir / "metrics" / "validation.json") or {}
    src_best = None
    src_final = None
    if isinstance(val.get("best"), dict):
        src_best = val["best"].get("accuracy")
    if isinstance(val.get("final"), dict):
        src_final = val["final"].get("accuracy")

    return CollapseReport(
        suite=suite, split_id=split_id, split_kind=split_kind,
        direction=direction, arm=arm, seed=seed, n_test=n,
        n_classes_in_task=int(n_classes_in_task),
        predicted_counts=counts,
        predicted_share=share,
        largest_predicted_class=(int(top_class) if top_class is not None else None),
        largest_predicted_share=float(top_share),
        entropy_bits=float(entropy),
        max_entropy_bits=float(max_ent),
        per_class_recall={int(k): float(v) for k, v in recalls.items()},
        per_class_precision={int(k): float(v) for k, v in precs.items()},
        observed_accuracy=float((y_true == y_pred).mean()),
        majority_baseline_accuracy=float(baseline["accuracy"]),
        majority_class=(
            int(baseline["majority_class"])
            if baseline["majority_class"] is not None else None
        ),
        accuracy_minus_majority=float(
            (y_true == y_pred).mean() - baseline["accuracy"]
        ),
        chance_level=(1.0 / n_classes_in_task) if n_classes_in_task > 0 else float("nan"),
        source_domain_best_val_accuracy=(
            float(src_best) if src_best is not None else None
        ),
        source_domain_final_val_accuracy=(
            float(src_final) if src_final is not None else None
        ),
        collapse_indicator={
            "top_class_share": ci.top_class_share,
            "n_classes_predicted": ci.n_classes_predicted,
            "entropy_bits": ci.entropy_bits,
            "max_entropy_bits": ci.max_entropy_bits,
            "top_share_threshold": ci.top_share_threshold,
            "min_classes_predicted_threshold": ci.min_classes_predicted_threshold,
            "collapsed": ci.collapsed,
        },
    )


def compute_feature_shift_stats(
    X_train: np.ndarray, X_target: np.ndarray,
) -> dict:
    """Feature-statistic shift between train and target folds.

    Any fitted component uses ``X_train`` only. We report:

    - per-dimension mean and std shift;
    - a summary norm of that shift;
    - a Mahalanobis-like z-score of X_target under N(mu_train, sigma_train).

    This is a diagnostic, never a decision function. Returned as plain
    Python floats/lists so it serialises to JSON.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    mu_tr = X_train.mean(axis=0)
    sd_tr = X_train.std(axis=0) + 1e-8
    mu_te = X_target.mean(axis=0)
    sd_te = X_target.std(axis=0) + 1e-8
    mean_shift = mu_te - mu_tr
    std_ratio = sd_te / sd_tr
    z = (X_target - mu_tr) / sd_tr
    per_sample_z_l2 = np.linalg.norm(z, axis=1)
    return {
        "n_train": int(X_train.shape[0]),
        "n_target": int(X_target.shape[0]),
        "dim": int(X_train.shape[1]) if X_train.ndim > 1 else 0,
        "mean_shift_l2": float(np.linalg.norm(mean_shift)),
        "mean_shift_max": float(np.max(np.abs(mean_shift))),
        "std_ratio_mean": float(np.mean(std_ratio)),
        "std_ratio_max": float(np.max(std_ratio)),
        "std_ratio_min": float(np.min(std_ratio)),
        "target_z_l2_mean": float(np.mean(per_sample_z_l2)),
        "target_z_l2_max": float(np.max(per_sample_z_l2)),
    }


def confidence_margin_stats(probs: np.ndarray | None) -> dict:
    """Top-1 probability + top1-top2 margin summary.

    Returns an empty dict when probabilities are unavailable (e.g. arm
    did not persist them).
    """
    if probs is None:
        return {}
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.size == 0:
        return {}
    sorted_p = np.sort(p, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2] if sorted_p.shape[1] >= 2 else np.zeros_like(top1)
    margin = top1 - top2
    return {
        "top1_mean": float(np.mean(top1)),
        "top1_median": float(np.median(top1)),
        "top1_p10": float(np.percentile(top1, 10)),
        "top1_p90": float(np.percentile(top1, 90)),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_p10": float(np.percentile(margin, 10)),
        "margin_p90": float(np.percentile(margin, 90)),
        "n_samples": int(p.shape[0]),
    }


def report_to_dict(rep: CollapseReport) -> dict:
    return {
        "suite": rep.suite,
        "split_id": rep.split_id,
        "split_kind": rep.split_kind,
        "direction": rep.direction,
        "arm": rep.arm,
        "seed": rep.seed,
        "n_test": rep.n_test,
        "n_classes_in_task": rep.n_classes_in_task,
        "predicted_counts": {str(k): int(v) for k, v in rep.predicted_counts.items()},
        "predicted_share": {str(k): float(v) for k, v in rep.predicted_share.items()},
        "largest_predicted_class": rep.largest_predicted_class,
        "largest_predicted_share": rep.largest_predicted_share,
        "entropy_bits": rep.entropy_bits,
        "max_entropy_bits": rep.max_entropy_bits,
        "per_class_recall": {str(k): float(v) for k, v in rep.per_class_recall.items()},
        "per_class_precision": {str(k): float(v) for k, v in rep.per_class_precision.items()},
        "observed_accuracy": rep.observed_accuracy,
        "majority_baseline_accuracy": rep.majority_baseline_accuracy,
        "majority_class": rep.majority_class,
        "accuracy_minus_majority": rep.accuracy_minus_majority,
        "chance_level": rep.chance_level,
        "source_domain_best_val_accuracy": rep.source_domain_best_val_accuracy,
        "source_domain_final_val_accuracy": rep.source_domain_final_val_accuracy,
        "collapse_indicator": rep.collapse_indicator,
    }
