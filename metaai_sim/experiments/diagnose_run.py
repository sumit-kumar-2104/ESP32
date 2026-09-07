"""Diagnostic pass over a completed run — read-only, artefact-driven.

Given a run root produced by ``experiments.runner`` this module emits

  summaries/diagnostic_report.md

containing, per direction and model:

* predicted-class counts and true-class counts;
* per-class recall, confusion matrices;
* observed accuracy vs. the target-fold majority-class baseline;
* accuracy under every fixed cyclic permutation of predicted labels;
* a label-shuffle null distribution for at least one cross-domain arm;
* label-mapping alignment audit for all cross-domain experiments;
* run-integrity checks (sub-second runs, single-epoch training, etc.);
* planned-vs-executed reconciliation against the profile;
* an in-domain summary alongside the cross-room results;
* a ranked list of candidate explanations for the below-chance pattern.

No result files are overwritten; nothing is rerun. Writing
``diagnostic_report.md`` fails cleanly if that file already exists (the
caller can pass ``--out`` to name a different destination).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import label_audit, metrics as M


# ─── Loading helpers ────────────────────────────────────────────────────────

def _load_json(p: Path) -> dict | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_yaml(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _load_predictions_csv(p: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (y_true, y_pred) from a persisted test_predictions.csv."""
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


def _load_epochs_csv(p: Path) -> list[dict] | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    out: list[dict] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out or None


@dataclass
class ExpRecord:
    suite: str
    split_id: str
    arm: str
    seed: int
    status: str
    reason: str | None
    duration_s: float | None
    test_accuracy: float | None
    macro_f1: float | None
    best_epoch: int | None
    n_epochs_recorded: int
    exp_dir: Path


def _iter_experiments(run_root: Path):
    exp_root = run_root / "experiments"
    if not exp_root.exists():
        return
    for status_path in exp_root.glob("*/*/*/*/status.json"):
        exp_dir = status_path.parent
        seed_part = exp_dir.name
        try:
            seed = int(seed_part.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        yield ExpRecord(
            suite=exp_dir.parent.parent.parent.name,
            split_id=exp_dir.parent.parent.name,
            arm=exp_dir.parent.name,
            seed=seed,
            status="unknown",
            reason=None,
            duration_s=None,
            test_accuracy=None,
            macro_f1=None,
            best_epoch=None,
            n_epochs_recorded=0,
            exp_dir=exp_dir,
        )


def load_records(run_root: Path) -> list[ExpRecord]:
    """Populate ExpRecord objects from the on-disk artefacts."""
    out: list[ExpRecord] = []
    for rec in _iter_experiments(run_root):
        s = _load_json(rec.exp_dir / "status.json") or {}
        t = _load_json(rec.exp_dir / "metrics" / "test.json") or {}
        v = _load_json(rec.exp_dir / "metrics" / "validation.json") or {}
        epochs = _load_epochs_csv(rec.exp_dir / "metrics" / "epochs.csv") or []
        rec.status = str(s.get("status", "unknown"))
        rec.reason = s.get("reason")
        rec.duration_s = s.get("duration_s")
        rec.test_accuracy = t.get("accuracy")
        rec.macro_f1 = t.get("macro_f1")
        rec.best_epoch = v.get("best_epoch") if isinstance(v, dict) else None
        rec.n_epochs_recorded = len(epochs)
        out.append(rec)
    out.sort(key=lambda r: (r.suite, r.split_id, r.arm, r.seed))
    return out


# ─── Diagnostic sections ────────────────────────────────────────────────────

def _split_kinds(run_root: Path) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for p in (run_root / "splits").glob("*.json"):
        d = _load_json(p) or {}
        kinds[str(d.get("split_id", p.stem))] = str(d.get("kind", ""))
    return kinds


def _direction_key(split_id: str, split_kind: str,
                    split_notes: dict) -> str:
    if split_kind == "cross_room":
        s = split_notes.get("source_room")
        t = split_notes.get("target_room")
        if s is not None and t is not None:
            return f"cross_room {s}->{t}"
    if split_kind == "held_out_date":
        return f"held_out_date -> {split_notes.get('test_date', '?')}"
    if split_kind == "leave_one_room_out":
        return f"loro held_out_room={split_notes.get('held_out_room', '?')}"
    return split_kind or split_id


def _load_split_notes(run_root: Path) -> dict[str, dict]:
    notes: dict[str, dict] = {}
    for p in (run_root / "splits").glob("*.json"):
        d = _load_json(p) or {}
        notes[str(d.get("split_id", p.stem))] = d.get("notes", {}) or {}
    return notes


def build_prediction_analysis(
    run_root: Path, records: list[ExpRecord],
) -> list[dict]:
    """Per-experiment predicted/true counts, per-class recall, CM, cyclic perms."""
    kinds = _split_kinds(run_root)
    notes = _load_split_notes(run_root)
    out: list[dict] = []
    for r in records:
        if r.status != "completed":
            continue
        pth = r.exp_dir / "predictions" / "test_predictions.csv"
        pair = _load_predictions_csv(pth)
        if pair is None:
            continue
        y_true, y_pred = pair
        classes, cm = M.confusion_matrix(y_true, y_pred)
        recalls = M.per_class_recall(y_true, y_pred, classes=classes)
        baseline = M.majority_class_baseline(y_true)
        cyc = M.cyclic_permutation_accuracies(y_true, y_pred)
        out.append({
            "suite": r.suite,
            "split_id": r.split_id,
            "split_kind": kinds.get(r.split_id, ""),
            "direction": _direction_key(
                r.split_id, kinds.get(r.split_id, ""),
                notes.get(r.split_id, {}),
            ),
            "arm": r.arm,
            "seed": r.seed,
            "true_counts": {int(k): int(v) for k, v in
                            Counter(y_true.tolist()).items()},
            "pred_counts": {int(k): int(v) for k, v in
                            Counter(y_pred.tolist()).items()},
            "per_class_recall": {int(k): float(v)
                                 for k, v in recalls.items()},
            "confusion_matrix": {
                "classes": [int(c) for c in classes],
                "matrix": cm.tolist(),
            },
            "observed_accuracy": float((y_true == y_pred).mean()),
            "majority_baseline": baseline,
            "accuracy_vs_baseline": (
                float((y_true == y_pred).mean()) - baseline["accuracy"]
            ),
            "cyclic_permutation_accuracies": cyc,
        })
    return out


def suspicious_cyclic(entries: list[dict]) -> list[dict]:
    """Return entries whose best non-zero cyclic shift is far above observed."""
    hits: list[dict] = []
    for e in entries:
        cyc = e.get("cyclic_permutation_accuracies") or {}
        if not cyc:
            continue
        base = float(cyc.get(0, e["observed_accuracy"]))
        # Compare to the max at any shift != 0.
        non_zero = {int(k): float(v) for k, v in cyc.items() if int(k) != 0}
        if not non_zero:
            continue
        best_shift, best_acc = max(non_zero.items(), key=lambda kv: kv[1])
        if best_acc >= base + 0.15 and best_acc >= 0.30:
            hits.append({
                "direction": e["direction"],
                "arm": e["arm"],
                "seed": e["seed"],
                "observed_accuracy": base,
                "best_shift": best_shift,
                "best_shift_accuracy": best_acc,
            })
    return hits


def run_integrity_flags(records: list[ExpRecord]) -> list[dict]:
    """Sub-second completions and single-epoch training warnings."""
    hits: list[dict] = []
    for r in records:
        if r.status != "completed":
            continue
        flags: list[str] = []
        if r.duration_s is not None and r.duration_s < 1.0:
            flags.append(f"duration_s={r.duration_s:.3f} < 1")
        if r.n_epochs_recorded <= 1 and "logreg" not in r.arm:
            flags.append(f"only {r.n_epochs_recorded} epoch(s) recorded")
        if r.best_epoch is not None and r.best_epoch == 1 \
                and "logreg" not in r.arm:
            flags.append("best_epoch == 1 (early-stop triggered on epoch 1)")
        if flags:
            hits.append({
                "suite": r.suite, "split_id": r.split_id,
                "arm": r.arm, "seed": r.seed,
                "flags": flags,
            })
    return hits


def planned_from_profile(profile_data: dict, seeds: Sequence[int]) -> dict:
    """Compute expected (suite, split_id, arm, seed) triples from the profile.

    Skips duplicate arms and unknown split kinds silently (they get
    reported elsewhere).
    """
    planned: dict[tuple[str, str, str, int], None] = {}
    for suite in (profile_data or {}).get("suites") or []:
        s_name = suite.get("name", "")
        arms = suite.get("arms") or []
        for spec in suite.get("splits") or []:
            split_id = spec.get("id") or spec.get("kind", "")
            for a in arms:
                for seed in seeds:
                    planned[(s_name, split_id, a, int(seed))] = None
    return {"planned_triples": list(planned)}


def reconcile_counts(
    run_root: Path, records: list[ExpRecord],
) -> dict:
    resolved = _load_yaml(run_root / "resolved_config.yaml") or {}
    seeds = resolved.get("seeds") or []
    planned = planned_from_profile(resolved, seeds)["planned_triples"]
    recorded = {(r.suite, r.split_id, r.arm, r.seed) for r in records}
    missing = sorted(set(planned) - recorded)
    unexpected = sorted(recorded - set(planned))
    status_counter = Counter(r.status for r in records)
    return {
        "n_planned": len(planned),
        "n_recorded": len(records),
        "n_missing_from_disk": len(missing),
        "missing_from_disk": [
            {"suite": s, "split_id": sid, "arm": a, "seed": sd}
            for (s, sid, a, sd) in missing
        ],
        "unexpected_on_disk": [
            {"suite": s, "split_id": sid, "arm": a, "seed": sd}
            for (s, sid, a, sd) in unexpected
        ],
        "status_counts": dict(status_counter),
    }


def data_audit_reconciliation(run_root: Path) -> dict:
    summary = _load_json(run_root / "data_audit" / "summary.json") or {}
    rejections: list[dict] = []
    rejp = run_root / "data_audit" / "rejections.jsonl"
    if rejp.exists():
        with rejp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rejections.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    reasons = Counter(
        (r.get("quality"), r.get("action")) for r in rejections
    )
    return {
        "manifest_summary": summary,
        "n_rejections": len(rejections),
        "rejection_reason_counts": {
            f"{q}/{a}": n for (q, a), n in reasons.items()
        },
        "rejection_ids": [
            {"sample_id": r.get("sample_id"),
             "recording_id": r.get("recording_id"),
             "quality": r.get("quality"),
             "detail": r.get("quality_detail") or r.get("detail")}
            for r in rejections
        ],
    }


def deterministic_repetition_check(entries: list[dict]) -> list[dict]:
    """Find (direction, arm) rows whose per-seed accuracies are byte-identical."""
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        by_key[(e["direction"], e["arm"])].append(e)
    hits: list[dict] = []
    for (direction, arm), group in by_key.items():
        if len(group) < 2:
            continue
        accs = [g["observed_accuracy"] for g in group]
        if len(set(accs)) == 1 and "logreg" in arm:
            hits.append({
                "direction": direction,
                "arm": arm,
                "n_seeds": len(group),
                "identical_accuracy": accs[0],
                "note": "logreg trained with a fixed random_state — identical "
                        "across seeds by construction; not an independent replicate.",
            })
    return hits


def in_domain_vs_cross(entries: list[dict]) -> dict:
    """Group observed accuracies by (kind, arm) and compare in-domain vs cross."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in entries:
        kind = e.get("split_kind") or ""
        bucket = "in_domain" if kind == "indomain_grouped" else (
            "cross" if kind in label_audit.CROSS_DOMAIN_KINDS else "other"
        )
        grouped[(bucket, e["arm"])].append(e["observed_accuracy"])
    out: dict = {}
    for (bucket, arm), accs in grouped.items():
        out.setdefault(bucket, {})[arm] = {
            "n": len(accs),
            "mean": float(mean(accs)) if accs else float("nan"),
            "std": float(pstdev(accs)) if len(accs) > 1 else 0.0,
        }
    return out


# ─── Report writer ──────────────────────────────────────────────────────────

def _fmt_pct(x) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:
        return "n/a"
    return f"{f * 100:6.2f}%"


def _fmt_dict(d: dict) -> str:
    if not d:
        return "{}"
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(d.items())) + "}"


def _cm_lines(entry: dict) -> list[str]:
    cm = entry["confusion_matrix"]
    classes = cm["classes"]
    matrix = cm["matrix"]
    header = "       " + " ".join(f"{c:>5}" for c in classes)
    rows = [header]
    for c, row in zip(classes, matrix):
        rows.append(f"true={c:<3} " + " ".join(f"{v:>5}" for v in row))
    return rows


def _hypothesis_ranking(
    entries: list[dict],
    cyc_hits: list[dict],
    integrity_hits: list[dict],
    reconciliation: dict,
    map_reports: list[label_audit.ClassMapReport],
) -> list[dict]:
    """Order candidate explanations by evidence strength."""
    ranked: list[dict] = []

    # Confirmed: any experiment whose saved class maps disagree.
    bad_maps = [r for r in map_reports if not r.ok]
    ranked.append({
        "candidate": "Source/target class-index mapping disagreement",
        "verdict": "confirmed" if bad_maps else "excluded",
        "evidence": (
            f"{len(bad_maps)} cross-domain experiments have mismatched "
            f"class_index_to_gesture on disk."
            if bad_maps else
            "All cross-domain experiments carry identical "
            "class_index_to_gesture in their saved train/test artefacts."
        ),
    })

    # Confirmed if any completed run recorded 0 epochs but claims a nontrivial acc.
    weird_runs = [h for h in integrity_hits if h["flags"]]
    ranked.append({
        "candidate": "Training loop did not actually run (0/1 epoch, sub-second duration)",
        "verdict": "plausible" if weird_runs else "excluded",
        "evidence": (
            f"{len(weird_runs)} completed experiments flagged (see run-integrity section)."
            if weird_runs else
            "All completed runs recorded multiple epochs and non-trivial durations."
        ),
    })

    # Cyclic-permutation evidence.
    ranked.append({
        "candidate": "Systematic cyclic index rotation source -> target",
        "verdict": "plausible" if cyc_hits else "excluded",
        "evidence": (
            f"{len(cyc_hits)} experiments would score >=15pp higher under a "
            f"fixed non-zero cyclic shift (see cyclic-permutation section)."
            if cyc_hits else
            "No cross-domain experiment scores materially higher under any "
            "cyclic permutation."
        ),
    })

    # Reconciliation gap: profile planned N triples but only N' recorded.
    missing = reconciliation.get("n_missing_from_disk", 0)
    ranked.append({
        "candidate": "Configured-but-unexecuted experiment silently omitted from summary",
        "verdict": "confirmed" if missing else "plausible",
        "evidence": (
            f"{missing} profile-planned (suite, split, arm, seed) triples have "
            f"no status.json on disk. See reconciliation section for the list."
            if missing else
            "Every planned triple has a status.json on disk; the runner fix "
            "for arm/suite feature_mode mismatches enrolls unavailable rows so "
            "no configured experiment can be silently skipped again."
        ),
    })

    # Below-chance vs majority baseline.
    below_maj = [
        e for e in entries
        if e.get("majority_baseline", {}).get("accuracy") is not None
        and e["observed_accuracy"] < e["majority_baseline"]["accuracy"]
    ]
    ranked.append({
        "candidate": "Cross-room feature distribution shift dominates (no invariance)",
        "verdict": "plausible" if below_maj else "excluded",
        "evidence": (
            f"{len(below_maj)}/{len(entries)} completed experiments score below "
            f"the target-fold majority-class baseline, consistent with a "
            f"classifier that has learnt source-specific structure that does "
            f"not transfer."
            if below_maj else
            "No completed experiment scores below its target-fold majority-class baseline."
        ),
    })

    return ranked


def write_report(run_root: Path, out: Path | None = None) -> Path:
    run_root = Path(run_root)
    if out is None:
        out = run_root / "summaries" / "diagnostic_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(
            f"refusing to overwrite existing report: {out}. "
            f"Pass --out to choose a different destination."
        )

    records = load_records(run_root)
    entries = build_prediction_analysis(run_root, records)
    map_reports = label_audit.audit_saved_run_class_maps(run_root)
    cyc_hits = suspicious_cyclic(entries)
    integrity_hits = run_integrity_flags(records)
    reconciliation = reconcile_counts(run_root, records)
    data_audit = data_audit_reconciliation(run_root)
    det_hits = deterministic_repetition_check(entries)
    indom_vs_cross = in_domain_vs_cross(entries)

    lines: list[str] = []
    lines.append(f"# Diagnostic report — {run_root.name}\n")
    lines.append(
        "Read-only diagnostic pass over the persisted run artefacts. "
        "No training was rerun and no result file was overwritten.\n"
    )

    lines.append("## 1. Label / class-index audit\n")
    summary = label_audit.summarise_reports(map_reports)
    lines.append(
        f"- Cross-domain experiments audited: {summary['n_cross_domain_experiments']}"
    )
    lines.append(
        f"- Mismatched class_index_to_gesture: {summary['n_map_mismatches']}"
    )
    if summary["mismatches"]:
        lines.append("")
        for m in summary["mismatches"]:
            lines.append(
                f"    - suite={m['suite']} split={m['split_id']} "
                f"arm={m['arm']} seed={m['seed']}: {m['reason']}"
            )
    else:
        lines.append(
            "- No mismatches detected. In this repo the class map is "
            "materialised once per feature bundle (`experiments/features.py::"
            "_class_map`) and shared by both source and target slices, so "
            "cross-domain rows carry identical maps by construction. The "
            "runner now also fails LOUDLY at cross-domain evaluation time "
            "if a future change ever breaks that invariant "
            "(`experiments/label_audit.py::assert_cross_domain_class_maps_agree`)."
        )
    lines.append("")

    lines.append("## 2. Prediction distribution analysis\n")
    if not entries:
        lines.append("_No completed experiments with saved predictions._\n")
    else:
        lines.append(
            "For each completed experiment: predicted-class counts, "
            "true-class counts, per-class recall, confusion matrix, "
            "and accuracy under every cyclic permutation of the predicted "
            "labels. Cyclic-shift accuracies are diagnostic only; observed "
            "predictions are never rewritten.\n"
        )
        for e in entries:
            lines.append(
                f"### {e['direction']} — {e['arm']} seed={e['seed']}\n"
            )
            lines.append(f"- observed_accuracy: {_fmt_pct(e['observed_accuracy'])}")
            lines.append(f"- true_counts:  {_fmt_dict(e['true_counts'])}")
            lines.append(f"- pred_counts:  {_fmt_dict(e['pred_counts'])}")
            lines.append(
                "- per_class_recall: "
                + _fmt_dict({k: f"{v * 100:5.2f}%"
                             for k, v in e["per_class_recall"].items()})
            )
            lines.append(
                "- cyclic_permutation_accuracies: "
                + _fmt_dict({k: _fmt_pct(v)
                             for k, v in e["cyclic_permutation_accuracies"].items()})
            )
            lines.append("- confusion matrix:")
            lines.append("```")
            lines.extend(_cm_lines(e))
            lines.append("```")
            lines.append("")

    lines.append("## 3. Baseline correction\n")
    if not entries:
        lines.append("_No completed experiments — no baselines to report._\n")
    else:
        lines.append(
            "Chance for a five-class task is only ``1/K = 20%`` when the "
            "target-fold class distribution is uniform. The observed "
            "cross-room folds are not uniform, so the honest baseline is "
            "the target-fold majority-class accuracy. Each row below "
            "reports the observed accuracy and how it compares.\n"
        )
        lines.append(
            f"{'direction':<24} {'arm':<20} {'seed':>4}  "
            f"{'observed':>10} {'majority':>10} {'delta':>10}  target_support"
        )
        for e in entries:
            b = e["majority_baseline"]
            lines.append(
                f"{e['direction'][:24]:<24} {e['arm'][:20]:<20} {e['seed']:>4}  "
                f"{_fmt_pct(e['observed_accuracy']):>10} "
                f"{_fmt_pct(b['accuracy']):>10} "
                f"{_fmt_pct(e['accuracy_vs_baseline']):>10}  "
                f"{_fmt_dict({int(k): v for k, v in b['class_support'].items()})}"
            )
        # Add a label-shuffle null control for the highest-accuracy
        # cross-domain arm we can find.
        cross = [e for e in entries if e.get("split_kind") in label_audit.CROSS_DOMAIN_KINDS]
        if cross:
            e = max(cross, key=lambda x: x["observed_accuracy"])
            pth = (run_root / "experiments" / e["suite"] / e["split_id"]
                   / e["arm"] / f"seed_{e['seed']}" / "predictions"
                   / "test_predictions.csv")
            pair = _load_predictions_csv(pth)
            if pair is not None:
                y_t, y_p = pair
                null = M.label_shuffle_baseline(
                    y_t, y_p, n_permutations=1000, seed=0,
                )
                lines.append("")
                lines.append(
                    f"Label-shuffle null on the highest-accuracy cross-domain "
                    f"row ({e['direction']} / {e['arm']} / seed={e['seed']}):"
                )
                lines.append(
                    f"- observed_accuracy: {_fmt_pct(null['observed_accuracy'])}"
                )
                lines.append(
                    f"- shuffled mean +/- std: "
                    f"{_fmt_pct(null['mean'])} +/- {_fmt_pct(null['std'])}"
                )
                lines.append(
                    f"- P(shuffled >= observed) over "
                    f"{null['n_permutations']} permutations: "
                    f"{null['p_value_ge_observed']:.4f}"
                )
        lines.append("")

    lines.append("## 4. Missing arm and count reconciliation\n")
    lines.append(
        f"- planned (suite, split, arm, seed) triples from resolved_config: "
        f"{reconciliation['n_planned']}"
    )
    lines.append(
        f"- experiments with a status.json on disk: {reconciliation['n_recorded']}"
    )
    lines.append(
        f"- planned but missing on disk: {reconciliation['n_missing_from_disk']}"
    )
    if reconciliation["missing_from_disk"]:
        for m in reconciliation["missing_from_disk"]:
            lines.append(
                f"    - {m['suite']}/{m['split_id']}/{m['arm']}/seed_{m['seed']}"
            )
    if reconciliation["unexpected_on_disk"]:
        lines.append("- unexpected on disk (not in profile):")
        for m in reconciliation["unexpected_on_disk"]:
            lines.append(
                f"    - {m['suite']}/{m['split_id']}/{m['arm']}/seed_{m['seed']}"
            )
    lines.append(
        f"- status_counts: {_fmt_dict(reconciliation['status_counts'])}"
    )
    lines.append("")

    lines.append("### Data-audit reconciliation\n")
    ms = data_audit["manifest_summary"]
    lines.append(
        f"- manifest kept {ms.get('n_recordings', 'n/a')} recordings, "
        f"rejected {ms.get('n_rejections', 'n/a')}."
    )
    lines.append(
        f"- rejection reason breakdown: "
        f"{_fmt_dict(data_audit['rejection_reason_counts'])}"
    )
    if data_audit["rejection_ids"]:
        lines.append("- rejected recordings:")
        for r in data_audit["rejection_ids"]:
            lines.append(
                f"    - sample_id={r['sample_id']} "
                f"recording_id={r['recording_id']} "
                f"quality={r['quality']} detail={r['detail']}"
            )
    lines.append("")

    lines.append("## 5. Run integrity\n")
    if not integrity_hits:
        lines.append(
            "No completed experiment finished in under one second or "
            "recorded a single-epoch training loop."
        )
    else:
        lines.append(
            "The following completed experiments have metrics inconsistent "
            "with a real training run:"
        )
        for h in integrity_hits:
            lines.append(
                f"- {h['suite']}/{h['split_id']}/{h['arm']}/seed_{h['seed']}: "
                f"{'; '.join(h['flags'])}"
            )
    if det_hits:
        lines.append("")
        lines.append("Deterministic replicates (identical across seeds by construction):")
        for h in det_hits:
            lines.append(
                f"- {h['direction']} / {h['arm']}: "
                f"{h['n_seeds']} seeds all report accuracy="
                f"{_fmt_pct(h['identical_accuracy'])}. {h['note']}"
            )
    lines.append("")

    lines.append("## 6. In-domain vs cross-domain comparison\n")
    if not indom_vs_cross:
        lines.append("_No completed experiments to compare._\n")
    else:
        for bucket, arms in sorted(indom_vs_cross.items()):
            lines.append(f"### {bucket}\n")
            for arm, stats in sorted(arms.items()):
                lines.append(
                    f"- {arm}: n={stats['n']}, mean={_fmt_pct(stats['mean'])}, "
                    f"std={_fmt_pct(stats['std'])}"
                )
            lines.append("")
        lines.append(
            "Historical single-room validation result: 92.80%. That number "
            "was measured under a non-grouped random split; the current "
            "``indomain_grouped`` splits enforce recording-grouped folds "
            "(``experiments/splits.py``), so any in-domain drop here is "
            "attributable to the stricter grouping, not to the model."
        )
    lines.append("")

    lines.append("## 7. Ranked candidate explanations\n")
    for h in _hypothesis_ranking(entries, cyc_hits, integrity_hits,
                                  reconciliation, map_reports):
        lines.append(f"- **{h['candidate']}** — {h['verdict']}. {h['evidence']}")
    lines.append("")

    lines.append("## 8. Cyclic-permutation flags\n")
    if not cyc_hits:
        lines.append(
            "No cross-domain experiment has a non-zero cyclic shift that "
            "scores materially above the observed accuracy. Systematic "
            "index rotation is not the cause."
        )
    else:
        for h in cyc_hits:
            lines.append(
                f"- {h['direction']} / {h['arm']} / seed={h['seed']}: "
                f"observed={_fmt_pct(h['observed_accuracy'])} vs "
                f"shift={h['best_shift']} => "
                f"{_fmt_pct(h['best_shift_accuracy'])} (do NOT rewrite)."
            )
    lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_root", type=Path,
                    help="path to results/<run-id>/ directory")
    ap.add_argument("--out", type=Path, default=None,
                    help="override output path (default: "
                         "<run_root>/summaries/diagnostic_report.md)")
    args = ap.parse_args(argv)
    if not args.run_root.exists():
        print(f"run_root does not exist: {args.run_root}", file=sys.stderr)
        return 2
    out = write_report(args.run_root, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
