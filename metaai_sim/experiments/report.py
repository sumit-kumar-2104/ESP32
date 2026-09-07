"""Single human-readable text summary of a run.

Writes ``results/<run-id>/RUN_SUMMARY.txt`` at the top of the run tree so
you never have to click through experiment folders to see what happened.
Everything in this file is computed from artefacts already on disk
(``run_manifest.json``, ``data_audit/summary.json``, per-experiment
``status.json`` and ``metrics/test.json``, ``splits/*.json``) — no fresh
numbers are invented, and if the underlying files are missing the row
just says ``n/a``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from experiments import paths as paths_mod
from experiments import status as status_mod


BAR = "=" * 88
DIV = "-" * 88


def _fmt_dt(iso: str | None) -> str:
    if not iso:
        return "n/a"
    return iso


def _fmt_duration(start: str | None, end: str | None) -> str:
    if not start or not end:
        return "n/a"
    try:
        dt0 = datetime.fromisoformat(start)
        dt1 = datetime.fromisoformat(end)
    except ValueError:
        return "n/a"
    secs = int((dt1 - dt0).total_seconds())
    if secs < 0:
        return "n/a"
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_pct(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:  # NaN
        return "n/a"
    return f"{f * 100:6.2f}%"


def _fmt_mean_std(vals: Sequence[float]) -> str:
    clean = [v for v in vals if v is not None and v == v]
    if not clean:
        return "n/a"
    m = mean(clean)
    s = pstdev(clean) if len(clean) > 1 else 0.0
    return f"{m * 100:6.2f}% +/- {s * 100:5.2f}%"


def _trunc(s: str, width: int) -> str:
    s = str(s or "")
    return s if len(s) <= width else s[: width - 1] + "~"


def _load_json(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _iter_experiment_dirs(layout: paths_mod.RunLayout):
    if not layout.experiments.exists():
        return
    for status_path in layout.experiments.glob("*/*/*/*/status.json"):
        exp_dir = status_path.parent
        seed_part = exp_dir.name
        arm = exp_dir.parent.name
        split_id = exp_dir.parent.parent.name
        suite = exp_dir.parent.parent.parent.name
        try:
            seed = int(seed_part.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        yield {
            "suite": suite, "split_id": split_id, "arm": arm, "seed": seed,
            "exp_dir": exp_dir, "status_path": status_path,
        }


def _load_splits(layout: paths_mod.RunLayout) -> list[dict]:
    out: list[dict] = []
    if not layout.splits.exists():
        return out
    for p in sorted(layout.splits.glob("*.json")):
        d = _load_json(p)
        if not d:
            continue
        out.append(d)
    return out


def _load_data_audit(layout: paths_mod.RunLayout) -> dict:
    return _load_json(layout.data_audit / "summary.json") or {}


def _load_run_manifest(layout: paths_mod.RunLayout) -> dict:
    return _load_json(layout.run_manifest) or {}


def _collect_rows(layout: paths_mod.RunLayout) -> list[dict]:
    rows: list[dict] = []
    for meta in _iter_experiment_dirs(layout):
        s = status_mod.read_status(meta["status_path"]) or {}
        test_json = _load_json(meta["exp_dir"] / "metrics" / "test.json") or {}
        val_json = _load_json(meta["exp_dir"] / "metrics" / "validation.json") or {}
        best_val = None
        best_epoch = None
        if isinstance(val_json.get("best"), dict):
            best_val = val_json["best"].get("accuracy")
            best_epoch = val_json.get("best_epoch")
        rows.append({
            "suite": meta["suite"], "split_id": meta["split_id"],
            "arm": meta["arm"], "seed": meta["seed"],
            "status": s.get("status", "unknown"),
            "reason": s.get("reason"),
            "test_accuracy": test_json.get("accuracy"),
            "test_macro_f1": test_json.get("macro_f1"),
            "test_balanced_accuracy": test_json.get("balanced_accuracy"),
            "best_val_accuracy": best_val,
            "best_epoch": best_epoch,
            "duration_s": s.get("duration_s"),
        })
    rows.sort(key=lambda r: (r["suite"], r["split_id"], r["arm"], r["seed"]))
    return rows


def _section(title: str) -> str:
    return f"\n{DIV}\n{title}\n{DIV}\n"


def _header_block(manifest: dict, layout: paths_mod.RunLayout) -> str:
    git = manifest.get("git") or {}
    device = manifest.get("device") or {}
    counts = manifest.get("counts") or {}
    top_status = status_mod.read_status(layout.status) or {}
    lines = [
        BAR,
        f"RUN SUMMARY - {manifest.get('run_id', layout.root.name)}",
        BAR,
        "",
        f"profile:       {manifest.get('profile', 'n/a')}",
        f"started_at:    {_fmt_dt(manifest.get('started_at'))}",
        f"ended_at:      {_fmt_dt(manifest.get('ended_at'))}",
        f"duration:      {_fmt_duration(manifest.get('started_at'), manifest.get('ended_at'))}",
        f"git commit:    {git.get('short_commit', 'n/a')}  "
        f"(branch={git.get('branch', 'n/a')}, dirty={git.get('dirty', 'n/a')})",
        f"device:        {device.get('selected', 'n/a')}"
        + (f"  ({device.get('cuda_device_name')})" if device.get('cuda_device_name') else ""),
        f"dataset dir:   {manifest.get('dataset_dir') or 'n/a'}",
        f"raw csi dir:   {manifest.get('raw_csi_dir') or 'n/a'}",
        "",
        f"OVERALL STATUS: {top_status.get('status', 'n/a')}"
        + (f"  ({top_status.get('reason')})" if top_status.get('reason') else ""),
        f"counts:  " + "  ".join(f"{k}={counts.get(k, 0)}" for k in
                                  ("completed", "failed", "unavailable",
                                   "pending", "running")),
    ]
    return "\n".join(lines) + "\n"


def _data_audit_block(audit: dict) -> str:
    lines = [_section("DATA AUDIT")]
    lines.append(f"recordings kept:      {audit.get('n_recordings', 'n/a')}")
    lines.append(f"rejections:           {audit.get('n_rejections', 'n/a')}  "
                 f"(see data_audit/rejections.jsonl)")
    qc = audit.get("quality_counts") or {}
    if qc:
        lines.append("quality summary:")
        for k in sorted(qc):
            lines.append(f"  {k:<22} {qc[k]}")
    lines.append(f"rooms seen:   {audit.get('rooms_seen')}")
    lines.append(f"dates seen:   {audit.get('dates_seen')}")
    lines.append(f"users seen:   {audit.get('users_seen')}")
    gestures = audit.get("gestures") or {}
    if gestures:
        lines.append(f"gestures per date:            {gestures.get('gestures_per_date')}")
        lines.append(f"gestures common across dates: {gestures.get('gestures_common_across_dates')}")
    lines.append(f"cache_id:     {audit.get('cache_id', 'n/a')}")
    return "\n".join(lines) + "\n"


def _splits_block(splits: list[dict]) -> str:
    lines = [_section("SPLITS")]
    if not splits:
        lines.append("(no splits persisted)")
        return "\n".join(lines) + "\n"
    lines.append(f"{'split_id':<32} {'kind':<20} {'train':>7} {'val':>7} {'test':>7}  notes")
    for s in splits:
        checks = s.get("checks") or {}
        notes = s.get("notes") or {}
        note_bits = []
        if notes.get("date"):
            note_bits.append(f"date={notes['date']}")
        if notes.get("train_dates"):
            note_bits.append(f"train_dates={notes['train_dates']}")
        if notes.get("test_date"):
            note_bits.append(f"test_date={notes['test_date']}")
        if notes.get("source_room") is not None:
            note_bits.append(f"src_room={notes['source_room']}")
        if notes.get("target_room") is not None:
            note_bits.append(f"tgt_room={notes['target_room']}")
        if notes.get("held_out_room") is not None:
            note_bits.append(f"held_out_room={notes['held_out_room']}")
        if notes.get("room_date_confound"):
            note_bits.append("room/date confounded")
        lines.append(
            f"{_trunc(s.get('split_id'), 32):<32} "
            f"{_trunc(s.get('kind'), 20):<20} "
            f"{checks.get('n_train', 0):>7} "
            f"{checks.get('n_val', 0):>7} "
            f"{checks.get('n_test', 0):>7}  "
            f"{', '.join(note_bits)}"
        )
    return "\n".join(lines) + "\n"


def _experiments_block(rows: list[dict]) -> str:
    lines = [_section("PER-EXPERIMENT RESULTS")]
    if not rows:
        lines.append("(no experiments recorded)")
        return "\n".join(lines) + "\n"
    header = (
        f"{'suite':<18} {'split_id':<28} {'arm':<20} {'seed':>4} "
        f"{'status':<11} {'test_acc':>10} {'macro_f1':>10} {'bal_acc':>10} "
        f"{'best_ep':>7} {'dur_s':>7}"
    )
    lines.append(header)
    for r in rows:
        lines.append(
            f"{_trunc(r['suite'], 18):<18} "
            f"{_trunc(r['split_id'], 28):<28} "
            f"{_trunc(r['arm'], 20):<20} "
            f"{r['seed']:>4} "
            f"{_trunc(r['status'], 11):<11} "
            f"{_fmt_pct(r['test_accuracy']):>10} "
            f"{_fmt_pct(r['test_macro_f1']):>10} "
            f"{_fmt_pct(r['test_balanced_accuracy']):>10} "
            f"{(r['best_epoch'] if r['best_epoch'] is not None else '-'):>7} "
            f"{(int(r['duration_s']) if r['duration_s'] else '-'):>7}"
        )
    return "\n".join(lines) + "\n"


def _aggregate_block(rows: list[dict]) -> str:
    lines = [_section("AGGREGATE  (mean +/- std over seeds; completed rows only)")]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] != "completed":
            continue
        grouped[(r["suite"], r["split_id"], r["arm"])].append(r)
    if not grouped:
        lines.append("(no completed experiments)")
        return "\n".join(lines) + "\n"
    lines.append(
        f"{'suite':<18} {'split_id':<28} {'arm':<20} {'n':>3}  "
        f"{'test_acc':<18}  {'macro_f1':<18}  {'bal_acc':<18}"
    )
    for key in sorted(grouped):
        suite, split_id, arm = key
        rs = grouped[key]
        accs = [r["test_accuracy"] for r in rs if r["test_accuracy"] is not None]
        f1s = [r["test_macro_f1"] for r in rs if r["test_macro_f1"] is not None]
        bas = [r["test_balanced_accuracy"] for r in rs if r["test_balanced_accuracy"] is not None]
        lines.append(
            f"{_trunc(suite, 18):<18} "
            f"{_trunc(split_id, 28):<28} "
            f"{_trunc(arm, 20):<20} "
            f"{len(rs):>3}  "
            f"{_fmt_mean_std(accs):<18}  "
            f"{_fmt_mean_std(f1s):<18}  "
            f"{_fmt_mean_std(bas):<18}"
        )
    return "\n".join(lines) + "\n"


def _issues_block(rows: list[dict]) -> str:
    bad = [r for r in rows if r["status"] not in ("completed",)]
    if not bad:
        return ""
    lines = [_section("UNAVAILABLE / FAILED")]
    for r in bad:
        lines.append(
            f"[{r['status']}] {r['suite']}/{r['split_id']}/{r['arm']}/seed_{r['seed']}"
        )
        if r.get("reason"):
            lines.append(f"    reason: {r['reason']}")
    return "\n".join(lines) + "\n"


def _notes_block() -> str:
    lines = [_section("INTERPRETATION NOTES")]
    lines.extend([
        "- Val is used for early stop + checkpoint selection; test set touched once",
        "  with the selected checkpoint. Numbers above are locked-test.",
        "- StandardScaler was fit on the training fold only (asserted at runtime).",
        "- Historical README numbers are NOT copied here. Every number is computed",
        "  from persisted predictions/test_predictions.csv.",
        "- Room and recording date are confounded in the packaged Widar3.0 tree",
        "  (20181109 = room 1, 20181118 = room 2). 'cross-room' is also cross-date.",
        "- Statistical equivalence / non-inferiority is not auto-emitted. Only",
        "  mean +/- std across seeds is reported; interpret with care.",
        BAR,
    ])
    return "\n".join(lines) + "\n"


def write_run_report(layout: paths_mod.RunLayout) -> Path:
    """Write ``RUN_SUMMARY.txt`` and return its path."""
    manifest = _load_run_manifest(layout)
    audit = _load_data_audit(layout)
    splits = _load_splits(layout)
    rows = _collect_rows(layout)

    parts = [
        _header_block(manifest, layout),
        _data_audit_block(audit),
        _splits_block(splits),
        _experiments_block(rows),
        _aggregate_block(rows),
        _issues_block(rows),
        _notes_block(),
    ]
    text = "".join(parts)
    out = layout.root / "RUN_SUMMARY.txt"
    out.write_text(text, encoding="utf-8")
    return out
