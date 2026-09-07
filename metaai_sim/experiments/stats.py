"""Statistical reporting corrections.

Replaces the previous selection-biased single-row null (which took the
highest-scoring cross-domain result and permuted only that one) with:

- Pre-specified **primary comparisons** per split (chosen before results
  are inspected — see :data:`PRIMARY_COMPARISONS`).
- **Shuffle nulls for every row**, plus multiple-comparison correction
  (Benjamini-Hochberg FDR *and* Holm-Bonferroni FWER), so the caller can
  choose the correction they want to report.
- A clearly labelled **maximum-statistic null** — the null distribution
  of the maximum accuracy across all rows under label shuffling. Only
  the maximum-statistic p-value can honestly compare against a
  post-hoc-selected row.
- **Effect sizes in accuracy points**, alongside p-values.
- Explicit labelling of selection-biased comparisons as *exploratory*.

Deterministic replicates (fixed-seed sklearn logreg) continue to count
as ONE measurement.

Guardrail: :func:`format_finding` refuses to describe an at-chance
result as significant regardless of p-value.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Pre-specified primary comparison per cross-domain split kind.
# Chosen *before* looking at results and recorded here in code so nobody
# can retroactively swap the pre-registered target.
PRIMARY_COMPARISONS: dict[str, dict] = {
    "cross_room": {
        "arm": "digital_mlp_raw",
        "seed": 42,
        "note": "Primary transfer arm; digital MLP tops the in-domain leaderboard.",
    },
    "held_out_date": {
        "arm": "digital_mlp_raw",
        "seed": 42,
        "note": "Primary transfer arm; digital MLP tops the in-domain leaderboard.",
    },
    "leave_one_room_out": {
        "arm": "digital_mlp_raw",
        "seed": 42,
        "note": "Primary transfer arm; digital MLP tops the in-domain leaderboard.",
    },
    "same_room_different_date": {
        "arm": "digital_mlp_raw",
        "seed": 42,
        "note": "Primary session-effect arm; matches cross-room primary for parity.",
    },
}


AT_CHANCE_TOLERANCE = 0.02  # +/- 2 accuracy points around 1/K
CHANCE_LEVEL_SIG_GUARD_NOTE = (
    "Result is within +/- {tol:.0f} accuracy points of 1/K. Never reported "
    "as a significant finding even when the shuffle p-value is small."
)


@dataclass(frozen=True)
class ShuffleNull:
    observed_accuracy: float
    null_mean: float
    null_std: float
    p_value_ge_observed: float
    n_permutations: int


def shuffle_null(
    y_true: np.ndarray, y_pred: np.ndarray, *,
    n_permutations: int = 1000, seed: int = 0,
) -> ShuffleNull:
    from experiments import metrics as M
    d = M.label_shuffle_baseline(
        y_true, y_pred, n_permutations=n_permutations, seed=seed,
    )
    return ShuffleNull(
        observed_accuracy=float(d["observed_accuracy"]),
        null_mean=float(d["mean"]),
        null_std=float(d["std"]),
        p_value_ge_observed=float(d["p_value_ge_observed"]),
        n_permutations=int(d["n_permutations"]),
    )


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """BH-adjusted p-values (Q values). Ties get the pooled BH treatment.

    Returns a list the same length as ``p_values``; NaN entries stay NaN.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = int(np.isfinite(p).sum())
    if m == 0:
        return [float("nan")] * len(p)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranked = p[order]
    adj = np.full_like(ranked, float("nan"))
    finite = np.isfinite(ranked)
    ranks = np.arange(1, len(ranked) + 1, dtype=np.float64)
    adj_finite = ranked[finite] * m / ranks[finite]
    # Enforce monotonic non-decreasing from the top.
    adj_finite = np.minimum.accumulate(adj_finite[::-1])[::-1]
    adj[finite] = np.minimum(adj_finite, 1.0)
    out = np.full_like(p, float("nan"))
    out[order] = adj
    return [float(x) for x in out]


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    m = int(np.isfinite(p).sum())
    if m == 0:
        return [float("nan")] * len(p)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranked = p[order]
    adj = np.full_like(ranked, float("nan"))
    counter = 0
    running_max = 0.0
    for i, val in enumerate(ranked):
        if not np.isfinite(val):
            continue
        counter += 1
        scaled = val * (m - counter + 1)
        running_max = max(running_max, min(scaled, 1.0))
        adj[i] = running_max
    out = np.full_like(p, float("nan"))
    out[order] = adj
    return [float(x) for x in out]


def max_statistic_null(
    rows: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict:
    """Null distribution of ``max_i accuracy_i`` under label shuffling.

    For each permutation, every row's predictions get its own shuffle;
    the sampled statistic is the maximum accuracy across all rows.

    Returns ``observed_max`` plus the null summary + one-sided p-value
    ``P(max_null >= observed_max)``.
    """
    if not rows:
        return {"observed_max": float("nan"), "null_mean": float("nan"),
                "null_std": float("nan"),
                "p_value_ge_observed": float("nan"),
                "n_permutations": 0}
    rng = np.random.default_rng(seed)
    observed = 0.0
    for y_t, y_p in rows:
        acc = float((y_t == y_p).mean()) if y_t.size else 0.0
        observed = max(observed, acc)
    null_max = np.empty(n_permutations, dtype=np.float64)
    caches = [(np.asarray(y_t, dtype=int), np.asarray(y_p, dtype=int).copy())
              for y_t, y_p in rows]
    for i in range(n_permutations):
        m = 0.0
        for y_t, y_p in caches:
            if y_p.size:
                rng.shuffle(y_p)
                acc = float((y_t == y_p).mean())
                if acc > m:
                    m = acc
        null_max[i] = m
    return {
        "observed_max": float(observed),
        "null_mean": float(null_max.mean()),
        "null_std": float(null_max.std()),
        "p_value_ge_observed": float((null_max >= observed).mean()),
        "n_permutations": int(n_permutations),
    }


def at_chance(observed_accuracy: float, chance_level: float,
              tol: float = AT_CHANCE_TOLERANCE) -> bool:
    return abs(observed_accuracy - chance_level) <= tol


def format_finding(
    label: str,
    observed_accuracy: float,
    chance_level: float,
    baseline_accuracy: float,
    p_value: float | None,
    *,
    is_selection_biased: bool = False,
    tol: float = AT_CHANCE_TOLERANCE,
) -> str:
    """Human-readable one-liner with the guardrails applied."""
    effect_vs_chance = observed_accuracy - chance_level
    effect_vs_maj = observed_accuracy - baseline_accuracy
    parts = [
        f"{label}: acc={observed_accuracy * 100:.2f}%",
        f"chance={chance_level * 100:.2f}%",
        f"majority={baseline_accuracy * 100:.2f}%",
        f"delta_vs_chance={effect_vs_chance * 100:+.2f}pp",
        f"delta_vs_majority={effect_vs_maj * 100:+.2f}pp",
    ]
    if p_value is not None:
        parts.append(f"p={p_value:.4f}")
    if is_selection_biased:
        parts.append("[EXPLORATORY — selection-biased]")
    if at_chance(observed_accuracy, chance_level, tol=tol):
        parts.append(CHANCE_LEVEL_SIG_GUARD_NOTE.format(tol=tol * 100))
    return " | ".join(parts)


def _load_predictions(p: Path) -> tuple[np.ndarray, np.ndarray] | None:
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


def load_rows_for_null(
    run_root: Path,
    filter_split_kinds: Iterable[str] = (),
) -> list[dict]:
    """Gather ``(y_true, y_pred, meta)`` for completed rows on disk."""
    import json
    kinds: dict[str, str] = {}
    for p in (run_root / "splits").glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        kinds[str(d.get("split_id", p.stem))] = str(d.get("kind", ""))
    keep = set(filter_split_kinds) or None
    exp_root = run_root / "experiments"
    if not exp_root.exists():
        return []
    out: list[dict] = []
    for pred_path in exp_root.glob("*/*/*/*/predictions/test_predictions.csv"):
        pair = _load_predictions(pred_path)
        if pair is None:
            continue
        exp_dir = pred_path.parent.parent
        arm = exp_dir.parent.name
        split_id = exp_dir.parent.parent.name
        suite = exp_dir.parent.parent.parent.name
        kind = kinds.get(split_id, "")
        if keep is not None and kind not in keep:
            continue
        try:
            seed = int(exp_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        out.append({
            "suite": suite, "split_id": split_id, "arm": arm, "seed": seed,
            "split_kind": kind, "y_true": pair[0], "y_pred": pair[1],
        })
    out.sort(key=lambda r: (r["suite"], r["split_id"], r["arm"], r["seed"]))
    return out
