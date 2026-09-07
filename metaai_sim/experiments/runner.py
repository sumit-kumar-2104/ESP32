"""Orchestrator: reads a profile YAML and dispatches suites/splits/arms/seeds.

Contract:

* ``bash scripts/run_all.sh --profile smoke|core|full --device auto``
* ``bash scripts/run_all.sh --resume <run-id>``

Guarantees:

* Every experiment writes its own status.json atomically; a crash never
  leaves anything looking ``completed`` that isn't.
* Required-suite failures cause a non-zero exit code.
* Optional/unavailable suites are prominent in the summary.
* Resume compares config hashes: incompatible config or split refuses to
  reuse an existing run root.
* No dataset download, no package install, no unbounded search — the
  experiment list is fully enumerated by the profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import arms as arms_mod
from experiments import features as features_mod
from experiments import manifest as manifest_mod
from experiments import metrics as metrics_mod
from experiments import paths as paths_mod
from experiments import provenance
from experiments import splits as splits_mod
from experiments import status as status_mod


# ─── Config loading ─────────────────────────────────────────────────────────

@dataclass
class ProfileSuite:
    name: str
    required: bool
    kind: str                     # "raw_amplitude" | "dfs_debug" | "domain_learning" | ...
    arms: list[str] = field(default_factory=list)
    splits: list[dict[str, Any]] = field(default_factory=list)
    feature_mode: str = "raw"
    dfs_bins: str = "full"


@dataclass
class Profile:
    name: str
    label: str
    use_synthetic: bool
    dates: list[str]
    seeds: list[int]
    suites: list[ProfileSuite]
    dataset_dir_env: str = "METAAI_DATA_DIR"
    raw_csi_env: str = "METAAI_RAW_CSI_DIR"
    quality_policy: str = "reject"
    min_packets: int = 64
    read_packet_counts: bool = False


def load_profile(path: Path) -> Profile:
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"profile {path} must be a mapping")
    suites = [
        ProfileSuite(
            name=s["name"], required=bool(s.get("required", True)),
            kind=s.get("kind", "raw_amplitude"),
            arms=list(s.get("arms", [])),
            splits=list(s.get("splits", [])),
            feature_mode=s.get("feature_mode", "raw"),
            dfs_bins=s.get("dfs_bins", "full"),
        )
        for s in data.get("suites", [])
    ]
    return Profile(
        name=data.get("name", Path(path).stem),
        label=data.get("label", ""),
        use_synthetic=bool(data.get("use_synthetic", False)),
        dates=list(data.get("dates", [])),
        seeds=list(data.get("seeds", [42, 43, 44])),
        suites=suites,
        dataset_dir_env=data.get("dataset_dir_env", "METAAI_DATA_DIR"),
        raw_csi_env=data.get("raw_csi_env", "METAAI_RAW_CSI_DIR"),
        quality_policy=data.get("quality_policy", "reject"),
        min_packets=int(data.get("min_packets", 64)),
        read_packet_counts=bool(data.get("read_packet_counts", False)),
    )


def _hash_config(obj: dict) -> str:
    canonical = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


# ─── Orchestration log ──────────────────────────────────────────────────────

class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *args) -> None:
        msg = " ".join(str(a) for a in args)
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ─── Feature building + scaling ─────────────────────────────────────────────

def _fit_scaler_train_only(X_train: np.ndarray):
    """Fit StandardScaler on X_train only. Verified at runtime."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_train)
    # Match the assertion used in train_raw_csi.py / train_ab_dfs.py.
    assert getattr(sc, "n_samples_seen_", -1) == len(X_train), (
        "StandardScaler was NOT fit on TRAIN-fold-only "
        f"(n_samples_seen_={getattr(sc, 'n_samples_seen_', None)} != "
        f"len(X_train)={len(X_train)})."
    )
    return sc


# ─── Suite execution ────────────────────────────────────────────────────────

def _make_splits(
    records: list[manifest_mod.RecordingRecord],
    suite: ProfileSuite,
    profile: Profile,
    splits_dir: Path,
    log: Log,
) -> tuple[list[tuple[str, splits_mod.Split]], list[dict]]:
    """Materialise splits; return (valid, errors). Errors carry the
    intended split_id + kind + reason so the runner can enrol placeholder
    ``unavailable`` rows and surface them in summaries.
    """
    out: list[tuple[str, splits_mod.Split]] = []
    errors: list[dict] = []
    for i, spec in enumerate(suite.splits):
        kind = spec.get("kind", "unknown")
        seed = int(spec.get("seed", profile.seeds[0]))
        # Fall back to a stable id when the profile did not name the split.
        split_id = spec.get("id") or f"{kind}_{i}"
        try:
            if kind == "indomain_grouped":
                split = splits_mod.indomain_grouped(
                    records, date=spec["date"],
                    val_frac=float(spec.get("val_frac", 0.15)),
                    test_frac=float(spec.get("test_frac", 0.15)),
                    seed=seed, split_id=split_id,
                )
            elif kind == "held_out_date":
                split = splits_mod.held_out_date(
                    records,
                    train_dates=spec["train_dates"],
                    test_date=spec["test_date"],
                    val_frac=float(spec.get("val_frac", 0.15)),
                    seed=seed, split_id=split_id,
                )
            elif kind == "cross_room":
                split = splits_mod.cross_room(
                    records,
                    source_room=int(spec["source_room"]),
                    target_room=int(spec["target_room"]),
                    val_frac=float(spec.get("val_frac", 0.15)),
                    seed=seed, split_id=split_id,
                )
            elif kind == "leave_one_room_out":
                split = splits_mod.leave_one_room_out(
                    records,
                    held_out_room=int(spec["held_out_room"]),
                    val_frac=float(spec.get("val_frac", 0.15)),
                    seed=seed, split_id=split_id,
                )
            else:
                reason = f"unknown split kind {kind!r}"
                log(f"[splits] SKIP {split_id}: {reason}")
                errors.append({"split_id": split_id, "kind": kind, "reason": reason})
                continue
        except splits_mod.InvalidSplitError as e:
            log(f"[splits] SKIP {kind} ({split_id}): {e}")
            errors.append({"split_id": split_id, "kind": kind, "reason": str(e)})
            continue
        split.write(splits_dir)
        out.append((split.split_id, split))
        log(f"[splits] wrote {split.split_id} "
            f"(train={len(split.train_ids)}, val={len(split.val_ids)}, "
            f"test={len(split.test_ids)})")
    return out, errors


def _build_feature_bundle(
    profile: Profile,
    records: list[manifest_mod.RecordingRecord],
    feature_mode: str,
    dfs_bins: str,
    log: Log,
) -> features_mod.FeatureBundle:
    if profile.use_synthetic:
        _, bundle = features_mod.synthetic_fixture()
        log(f"[features] SYNTHETIC bundle: X={bundle.X.shape}")
        return bundle
    log(f"[features] building feature_mode={feature_mode} dfs_bins={dfs_bins} "
        f"n_records={len(records)}")
    bundle = features_mod.build_features_from_manifest(
        records, feature_mode=feature_mode, dfs_bins=dfs_bins,
    )
    log(f"[features] built X={bundle.X.shape}")
    return bundle


# ─── Runner ─────────────────────────────────────────────────────────────────

def _write_resolved_config(
    layout: paths_mod.RunLayout, profile: Profile, args: argparse.Namespace,
) -> None:
    import yaml
    resolved = {
        "profile": profile.name,
        "label": profile.label,
        "use_synthetic": profile.use_synthetic,
        "dates": profile.dates,
        "seeds": profile.seeds,
        "device": args.device,
        "results_root": str(args.results_root),
        "quality_policy": profile.quality_policy,
        "min_packets": profile.min_packets,
        "read_packet_counts": profile.read_packet_counts,
        "suites": [asdict(s) for s in profile.suites],
    }
    layout.resolved_config.write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8",
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _write_summaries(layout: paths_mod.RunLayout, index_rows: list[dict]) -> None:
    import csv
    layout.summaries.mkdir(parents=True, exist_ok=True)
    idx_path = layout.summaries / "experiment_index.csv"
    if not index_rows:
        idx_path.write_text("suite,split_id,arm,seed,status,reason\n", encoding="utf-8")
        (layout.summaries / "aggregate_metrics.csv").write_text(
            "arm,split_id,n_seeds,mean_test_acc,std_test_acc,mean_macro_f1\n",
            encoding="utf-8",
        )
        (layout.summaries / "findings.md").write_text(
            "# Findings\n\nNo experiments recorded.\n", encoding="utf-8",
        )
        return
    fields = sorted({k for row in index_rows for k in row})
    with idx_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in index_rows:
            w.writerow(row)

    # Aggregate across seeds per (arm, split).
    agg: dict[tuple[str, str], list[dict]] = {}
    for row in index_rows:
        if row.get("status") != "completed":
            continue
        key = (row["arm"], row["split_id"])
        agg.setdefault(key, []).append(row)
    agg_path = layout.summaries / "aggregate_metrics.csv"
    with agg_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "split_id", "n_seeds", "mean_test_acc",
                    "std_test_acc", "mean_macro_f1"])
        for (arm, sid), rows in sorted(agg.items()):
            accs = np.array([float(r.get("test_accuracy", np.nan)) for r in rows])
            f1s = np.array([float(r.get("test_macro_f1", np.nan)) for r in rows])
            w.writerow([
                arm, sid, len(rows),
                float(np.nanmean(accs)) if accs.size else float("nan"),
                float(np.nanstd(accs)) if accs.size else float("nan"),
                float(np.nanmean(f1s)) if f1s.size else float("nan"),
            ])

    lines = ["# Findings", ""]
    counter: dict[str, int] = {}
    for row in index_rows:
        counter[row["status"]] = counter.get(row["status"], 0) + 1
    lines.append("## Experiment counts by status")
    for k, v in sorted(counter.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Numbers here are computed only from persisted "
                 "`predictions/test_predictions.csv` files.")
    lines.append("- Historical results in README.md are NOT copied into "
                 "this file.")
    (layout.summaries / "findings.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def _run_suite(
    suite: ProfileSuite,
    profile: Profile,
    records: list[manifest_mod.RecordingRecord],
    layout: paths_mod.RunLayout,
    seeds: list[int],
    device_pref: str,
    log: Log,
    force: bool,
) -> tuple[list[dict], int]:
    """Return (index_rows, n_required_missing).

    ``n_required_missing`` counts everything in this suite that did NOT
    reach ``completed`` — but only when the suite is marked required. It
    is the number the top-level exit code depends on, so a split-level
    skip in a required suite propagates the same way as an arm-level
    failure.
    """
    rows: list[dict] = []

    concrete_splits, split_errors = _make_splits(
        records, suite, profile, layout.splits, log,
    )

    # Every skipped split still produces one ``unavailable`` row per
    # (arm, seed) so summaries and status counters reflect reality.
    for err in split_errors:
        reason = f"split unavailable ({err['kind']}): {err['reason']}"
        for arm_name in suite.arms:
            for seed in seeds:
                exp_dir = layout.experiment_dir(
                    suite.name, err["split_id"], arm_name, seed,
                )
                exp_dir.mkdir(parents=True, exist_ok=True)
                status_mod.write_status(
                    exp_dir / "status.json",
                    status="unavailable", reason=reason,
                )
                rows.append({
                    "suite": suite.name, "split_id": err["split_id"],
                    "arm": arm_name, "seed": seed,
                    "status": "unavailable", "reason": reason,
                })

    if not concrete_splits:
        req = "required" if suite.required else "optional"
        log(f"[suite:{suite.name}] no valid splits ({req}) — "
            f"{len(rows)} experiments marked unavailable")
        return rows, (len(rows) if suite.required else 0)

    bundle = _build_feature_bundle(
        profile, records, suite.feature_mode, suite.dfs_bins, log,
    )
    n_failed = 0

    for split_id, split in concrete_splits:
        try:
            X_tr, y_tr, sids_tr = features_mod.slice_bundle(bundle, split.train_ids)
            X_va, y_va, _ = features_mod.slice_bundle(bundle, split.val_ids)
            X_te, y_te, sids_te = features_mod.slice_bundle(bundle, split.test_ids)
        except KeyError as e:
            log(f"[suite:{suite.name}][{split_id}] cannot slice bundle: {e}")
            for arm_name in suite.arms:
                for seed in seeds:
                    exp_dir = layout.experiment_dir(suite.name, split_id, arm_name, seed)
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    status_mod.write_status(
                        exp_dir / "status.json",
                        status="unavailable",
                        reason=f"feature bundle missing sample_ids for split {split_id}",
                    )
                    rows.append({
                        "suite": suite.name, "split_id": split_id,
                        "arm": arm_name, "seed": seed, "status": "unavailable",
                        "reason": "feature bundle missing sample_ids",
                    })
            continue

        sc = _fit_scaler_train_only(X_tr)
        X_tr_s = sc.transform(X_tr).astype(np.float32)
        X_va_s = sc.transform(X_va).astype(np.float32)
        X_te_s = sc.transform(X_te).astype(np.float32)
        n_classes = int(np.unique(bundle.y).size)

        for arm_name in suite.arms:
            spec = arms_mod.get_arm(arm_name)
            # Match arm and suite feature intent so nothing is mis-wired.
            # Synthetic bundles bypass the check — the fixture is intentionally
            # feature-agnostic so we can exercise every arm's plumbing.
            if not bundle.is_synthetic and spec.feature_mode != suite.feature_mode:
                log(f"[suite:{suite.name}][{split_id}][{arm_name}] "
                    f"arm feature_mode={spec.feature_mode!r} != suite "
                    f"feature_mode={suite.feature_mode!r} — skipping arm.")
                continue

            for seed in seeds:
                exp_dir = layout.experiment_dir(suite.name, split_id, arm_name, seed)
                paths_mod.ensure_experiment_dir(exp_dir)
                status_path = exp_dir / "status.json"

                if status_mod.is_completed(status_path) and not force:
                    prev = status_mod.read_status(status_path)
                    log(f"[resume] SKIP already-completed {suite.name}/{split_id}/{arm_name}/seed_{seed}")
                    rows.append({
                        "suite": suite.name, "split_id": split_id,
                        "arm": arm_name, "seed": seed,
                        "status": "completed",
                        "test_accuracy": (prev or {}).get("test_accuracy"),
                        "test_macro_f1": (prev or {}).get("test_macro_f1"),
                        "test_balanced_accuracy": (prev or {}).get("test_balanced_accuracy"),
                    })
                    continue

                config_payload = {
                    "arm": asdict(spec),
                    "split_id": split_id,
                    "seed": seed,
                    "feature_mode": suite.feature_mode,
                    "dfs_bins": suite.dfs_bins,
                    "n_train": len(sids_tr), "n_val": len(X_va_s),
                    "n_test": len(sids_te),
                    "synthetic": bundle.is_synthetic,
                }
                (exp_dir / "config.yaml").write_text(
                    json.dumps(config_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                status_mod.write_status(status_path, status="running")
                _seed_all(seed)
                t0 = time.time()
                try:
                    summary = arms_mod.run_arm(
                        spec,
                        X_train=X_tr_s, y_train=y_tr,
                        X_val=X_va_s, y_val=y_va,
                        X_test=X_te_s, y_test=y_te,
                        sample_ids_test=sids_te,
                        seed=seed, n_classes=n_classes,
                        device_pref=device_pref,
                        exp_dir=exp_dir,
                        class_index_to_gesture=bundle.class_index_to_gesture,
                    )
                except Exception as e:  # noqa: BLE001 — arm-specific
                    trace = traceback.format_exc()
                    (exp_dir / "logs" / "console.log").parent.mkdir(parents=True, exist_ok=True)
                    (exp_dir / "logs" / "console.log").write_text(trace, encoding="utf-8")
                    status_mod.write_status(
                        status_path, status="failed",
                        reason=f"{type(e).__name__}: {e}",
                        extra={"duration_s": time.time() - t0},
                    )
                    log(f"[fail] {suite.name}/{split_id}/{arm_name}/seed_{seed}: {e}")
                    rows.append({
                        "suite": suite.name, "split_id": split_id,
                        "arm": arm_name, "seed": seed, "status": "failed",
                        "reason": f"{type(e).__name__}: {e}",
                    })
                    n_failed += 1
                    continue

                extra: dict[str, Any] = {"duration_s": time.time() - t0}
                extra.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str))})
                status_mod.write_status(
                    status_path, status="completed", extra=extra,
                )
                rows.append({
                    "suite": suite.name, "split_id": split_id,
                    "arm": arm_name, "seed": seed, "status": "completed",
                    **{k: v for k, v in summary.items() if isinstance(v, (int, float, str))},
                })
                log(f"[ok] {suite.name}/{split_id}/{arm_name}/seed_{seed}: "
                    f"test_acc={summary.get('test_accuracy'):.4f}")
    # For required suites, count both arm-level failures and any
    # split-level unavailable rows we enrolled above.
    if suite.required:
        n_missing = sum(1 for r in rows if r["status"] != "completed")
        return rows, n_missing
    return rows, n_failed


def _build_manifest(
    profile: Profile,
    layout: paths_mod.RunLayout,
    log: Log,
) -> list[manifest_mod.RecordingRecord]:
    if profile.use_synthetic:
        records, _ = features_mod.synthetic_fixture()
        manifest_mod.write_manifest(
            layout.data_audit, records, rejections=[],
            feature_mode="SYNTHETIC_FIXTURE", dfs_bins="n/a",
        )
        log(f"[manifest] SYNTHETIC: {len(records)} recordings")
        return records
    from config import get_raw_csi_dir
    csi_root = get_raw_csi_dir()
    if not csi_root.exists() or not any(csi_root.iterdir()):
        raise RuntimeError(
            f"raw CSI directory missing or empty: {csi_root}. "
            f"Set METAAI_RAW_CSI_DIR or METAAI_DATA_DIR."
        )
    log(f"[manifest] walking {csi_root} for dates={profile.dates}")
    records = manifest_mod.walk_raw_csi(
        csi_root, profile.dates,
        read_packet_counts=profile.read_packet_counts,
        min_packets=profile.min_packets,
    )
    kept, rejections = manifest_mod.apply_quality_policy(
        records, policy=profile.quality_policy,
    )
    summary = manifest_mod.write_manifest(
        layout.data_audit, kept, rejections,
        feature_mode="mixed", dfs_bins="mixed",
    )
    log(f"[manifest] kept={summary['n_recordings']} rejected={summary['n_rejections']}")
    return kept


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MetaAI evaluation orchestrator")
    ap.add_argument("--profile", type=str, default=None,
                    help="profile YAML under experiments/profiles/*.yaml, or a full path")
    ap.add_argument("--resume", type=str, default=None,
                    help="run-id under results/ to resume (must exist)")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "gpu"])
    ap.add_argument("--results-root", type=Path,
                    default=paths_mod.DEFAULT_RESULTS_ROOT)
    ap.add_argument("--force", action="store_true",
                    help="re-run experiments even if status.json says completed")
    ap.add_argument("--run-id", type=str, default=None,
                    help="override auto-generated run id (must match [A-Za-z0-9_.-]+)")
    args = ap.parse_args(argv)

    if args.resume and args.profile:
        raise SystemExit("choose --profile OR --resume, not both")
    if not args.resume and not args.profile:
        raise SystemExit("must give --profile <name-or-path> or --resume <run-id>")

    if args.resume:
        run_id = args.resume
        layout = paths_mod.build_run_layout(run_id, args.results_root)
        if not layout.run_manifest.exists():
            raise SystemExit(f"cannot resume — {layout.run_manifest} not found")
        manifest_before = json.loads(layout.run_manifest.read_text(encoding="utf-8"))
        profile_name = manifest_before.get("profile")
        profile_path = _resolve_profile_path(profile_name)
        profile = load_profile(profile_path)
    else:
        profile_path = _resolve_profile_path(args.profile)
        profile = load_profile(profile_path)
        run_id = args.run_id or _gen_run_id(profile.name)
        layout = paths_mod.build_run_layout(run_id, args.results_root)

    layout.ensure()
    log = Log(layout.orchestration_log)
    log(f"[run] run_id={run_id}   profile={profile.name}   "
        f"device={args.device}   force={args.force}")
    if profile.use_synthetic:
        log("[run] LABEL: synthetic fixture — NOT scientific results.")

    _write_resolved_config(layout, profile, args)
    provenance.write_environment_txt(layout.environment)

    # Resume config check: refuse if resolved_config diverges materially.
    if args.resume:
        prev_manifest = json.loads(layout.run_manifest.read_text(encoding="utf-8"))
        new_hash = _hash_config(_yaml_load(layout.resolved_config))
        prev_hash = prev_manifest.get("resolved_config_hash")
        if prev_hash and prev_hash != new_hash:
            raise SystemExit(
                "resume refused: resolved_config.yaml hash changed "
                f"({prev_hash} -> {new_hash})"
            )

    resolved_hash = _hash_config(_yaml_load(layout.resolved_config))
    manifest = provenance.RunManifest(
        run_id=run_id, profile=profile.name,
        launch_cmd=list(sys.argv),
        seeds=list(profile.seeds),
        models=sorted({a for s in profile.suites for a in s.arms}),
        suites=[s.name for s in profile.suites],
        dataset_dir=os.environ.get("METAAI_DATA_DIR"),
        raw_csi_dir=os.environ.get("METAAI_RAW_CSI_DIR"),
        git=provenance.git_info(REPO_ROOT),
        device=provenance.device_info(),
        versions=provenance.dependency_versions(),
        env=provenance.safe_env_snapshot(),
        constraints={
            "quality_policy": profile.quality_policy,
            "min_packets": profile.min_packets,
        },
    )
    # Add config hash so resume can detect divergence.
    payload = asdict(manifest)
    payload["resolved_config_hash"] = resolved_hash
    layout.run_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    status_mod.write_status(layout.status, status="running")

    try:
        records = _build_manifest(profile, layout, log)
    except Exception as e:  # noqa: BLE001
        log(f"[fatal] cannot build data manifest: {e}")
        status_mod.write_status(layout.status, status="failed",
                                 reason=f"manifest: {e}")
        return 2

    all_rows: list[dict] = []
    total_failed = 0
    total_required_failed = 0
    for suite in profile.suites:
        log(f"[suite:{suite.name}] required={suite.required} arms={suite.arms}")
        rows, n_failed = _run_suite(
            suite, profile, records, layout,
            seeds=list(profile.seeds), device_pref=args.device,
            log=log, force=args.force,
        )
        all_rows.extend(rows)
        total_failed += n_failed
        if suite.required:
            total_required_failed += n_failed

    _write_summaries(layout, all_rows)

    counts = {"pending": 0, "running": 0, "completed": 0,
              "failed": 0, "unavailable": 0}
    for r in all_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    payload = json.loads(layout.run_manifest.read_text(encoding="utf-8"))
    payload["counts"] = counts
    payload["ended_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    layout.run_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )

    log(f"[summary] {counts}")
    if total_required_failed:
        status_mod.write_status(
            layout.status, status="failed",
            reason=f"{total_required_failed} required-suite failures",
            extra={"counts": counts},
        )
        return 1
    status_mod.write_status(layout.status, status="completed",
                             extra={"counts": counts})
    return 0


def _yaml_load(path: Path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve_profile_path(name_or_path: str) -> Path:
    """Accept a bare profile name (smoke/core/full) or a full path."""
    p = Path(name_or_path)
    if p.exists():
        return p
    default = Path(__file__).parent / "profiles" / f"{name_or_path}.yaml"
    if default.exists():
        return default
    raise SystemExit(f"cannot find profile {name_or_path!r} (looked at {p} and {default})")


def _gen_run_id(profile_name: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = uuid.uuid4().hex[:6]
    return f"{ts}-{profile_name}-{tag}"


if __name__ == "__main__":
    raise SystemExit(run())
