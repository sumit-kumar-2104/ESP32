"""Reproduction report — CNN-GRU on structure-preserving tensors.

Reads persisted status.json + metrics/test.json under
``results/<run-id>/experiments/`` and writes
``summaries/reproduction_report.md`` with one section per
representation: achieved accuracy per split (in-domain, cross-room,
held-out-date), the Widar3.0 published reference, and the honest gap.

Never invents numbers — everything comes from disk. Every reference
value has an inline comment naming its source (Zheng et al., MobiSys
2019, Section 6.4 + Figure 19).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

from experiments import arms as arms_mod
from experiments import paths as paths_mod
from experiments import status as status_mod


# Published references, per feature representation. IN-DOMAIN column.
# Sourced from Widar3.0 (Zheng et al., MobiSys 2019, Fig 19 + Sec 6.4).
PUBLISHED_REFERENCES: dict[str, dict] = {
    "csi_tensor": {
        "in_domain": 40.2,
        "note": "Raw CSI, in-domain — Widar3.0 Fig 19.",
    },
    "dfs_tensor": {
        "in_domain": 77.8,
        "note": "DFS, in-domain — Widar3.0 Fig 19.",
    },
    "bvp_tensor": {
        "in_domain": 92.4,
        "note": "BVP, in-domain — Widar3.0 Fig 19.",
    },
}


def _fmt_pct(v) -> str:
    try:
        return f"{100.0 * float(v):6.2f}%"
    except (TypeError, ValueError):
        return "   n/a"


def _fmt_mean_std(vals: Sequence[float]) -> str:
    clean = [v for v in vals if v is not None and v == v]
    if not clean:
        return "n/a"
    m = mean(clean)
    s = pstdev(clean) if len(clean) > 1 else 0.0
    return f"{100 * m:5.2f}% +/- {100 * s:4.2f}%"


def _collect_rows(layout: paths_mod.RunLayout) -> list[dict]:
    """One row per ``(suite, split, arm, seed)`` under the run tree."""
    rows: list[dict] = []
    if not layout.experiments.exists():
        return rows
    for status_path in layout.experiments.glob("*/*/*/seed_*/status.json"):
        s = status_mod.read_status(status_path) or {}
        arm = status_path.parent.parent.name
        split = status_path.parent.parent.parent.name
        suite = status_path.parent.parent.parent.parent.name
        seed_dir = status_path.parent.name
        try:
            seed = int(seed_dir.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        test_json = None
        tp = status_path.parent / "metrics" / "test.json"
        if tp.exists():
            try:
                test_json = json.loads(tp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                test_json = None
        rows.append({
            "suite": suite, "split_id": split, "arm": arm, "seed": seed,
            "status": s.get("status"),
            "test_accuracy": (test_json or {}).get("accuracy"),
        })
    return rows


def _baseline_for_split(layout: paths_mod.RunLayout, split_id: str) -> dict | None:
    p = layout.summaries / "per_split_baselines" / f"{split_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _kind_label(kind: str) -> str:
    return {
        "indomain_grouped": "in-domain",
        "held_out_date": "held-out-date",
        "cross_room": "cross-room",
        "leave_one_room_out": "leave-one-room-out",
        "same_room_different_date": "same-room-different-date",
    }.get(kind, kind)


def write_reproduction_report(layout: paths_mod.RunLayout) -> Path:
    """Write ``summaries/reproduction_report.md`` and return its path."""
    rows = _collect_rows(layout)
    # Bucket completed cnn_gru rows by (feature_mode, split_id).
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    split_kinds: dict[str, str] = {}
    for r in rows:
        if r["status"] != "completed" or not r["arm"].startswith("cnn_gru"):
            continue
        try:
            spec = arms_mod.get_arm(r["arm"])
        except KeyError:
            continue
        # Only the plain cnn_gru arm counts for reproduction; constraint
        # variants are the ablation, not the reproduction gate.
        if spec.constraint is not None:
            continue
        # split kind lookup
        sp = layout.splits / f"{r['split_id']}.json"
        if sp.exists():
            try:
                split_kinds[r["split_id"]] = json.loads(
                    sp.read_text(encoding="utf-8")
                ).get("kind", "unknown")
            except json.JSONDecodeError:
                pass
        buckets[(spec.feature_mode, r["split_id"])].append(r)

    lines: list[str] = []
    lines.append("# Reproduction report — Widar3.0 CNN-GRU on tensor features")
    lines.append("")
    lines.append(
        "This report is auto-generated from persisted "
        "`metrics/test.json` files under `experiments/`. "
        "Every reference value is the published Widar3.0 number "
        "(Zheng et al., MobiSys 2019, Section 6.4 + Figure 19), "
        "sourced verbatim without adjustment."
    )
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    lines.append(
        "- Cross-room 2->1 in the packaged tree trains on ~1,049-1,259 "
        "samples and is data-scarce; it is reported but is NOT the "
        "primary comparison."
    )
    lines.append(
        "- 80% of the published figure is treated as an acceptable "
        "outcome; shortfalls are stated, not hidden."
    )
    lines.append(
        "- No hyperparameter was tuned against the target numbers."
    )
    lines.append("")

    for mode in ("csi_tensor", "dfs_tensor", "bvp_tensor"):
        ref = PUBLISHED_REFERENCES.get(mode, {})
        ref_in_domain = ref.get("in_domain")
        lines.append(f"## {mode}")
        lines.append("")
        lines.append(f"- published in-domain reference: **{ref_in_domain}%** — {ref.get('note', '')}")
        # Any splits present for this mode?
        mode_splits = sorted({sid for (fm, sid), _ in buckets.items() if fm == mode})
        if not mode_splits:
            lines.append("")
            lines.append(
                "- no completed cnn_gru rows for this representation. "
                "If the BVP source tree is missing, the `bvp_tensor` "
                "arm is marked `unavailable` at runtime with the exact "
                "reason; check the run manifest."
            )
            lines.append("")
            continue
        lines.append("")
        lines.append(
            "| split | kind | n seeds | test_acc (mean +/- std) | vs published in-domain |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for sid in mode_splits:
            group = buckets.get((mode, sid), [])
            accs = [r["test_accuracy"] for r in group if r["test_accuracy"] is not None]
            kind = split_kinds.get(sid, "?")
            gap_col = ""
            if ref_in_domain is not None and accs and kind == "indomain_grouped":
                achieved = 100 * mean(accs)
                gap = achieved - ref_in_domain
                pct_of_ref = achieved / ref_in_domain
                gate = "REACHED >= 80%" if pct_of_ref >= 0.80 else "SHORTFALL"
                gap_col = f"{achieved:5.2f}% (gap={gap:+.2f}pp, {gate})"
            elif kind != "indomain_grouped":
                gap_col = "cross-domain, no in-domain reference applies"
            lines.append(
                f"| `{sid}` | {_kind_label(kind)} | {len(accs)} | "
                f"{_fmt_mean_std(accs)} | {gap_col} |"
            )
        # Chance / majority baselines for context.
        lines.append("")
        for sid in mode_splits:
            b = _baseline_for_split(layout, sid) or {}
            base = b.get("baseline") or {}
            if base:
                lines.append(
                    f"- `{sid}` baselines: chance={_fmt_pct(base.get('chance_level'))}, "
                    f"majority_test={_fmt_pct(base.get('majority_class_accuracy'))} "
                    f"({base.get('majority_class_note') or ''})"
                )
        lines.append("")

    out = layout.summaries / "reproduction_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
