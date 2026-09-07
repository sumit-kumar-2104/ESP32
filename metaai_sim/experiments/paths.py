"""Canonical results/<run-id>/ layout.

Keeps every experiment collision-safe:

    results/<run-id>/
      run_manifest.json
      resolved_config.yaml
      environment.txt
      status.json
      logs/orchestration.log
      data_audit/
      splits/
      experiments/<suite>/<split-id>/<model>/seed_<seed>/
        config.yaml
        status.json
        logs/console.log
        logs/events.jsonl
        metrics/epochs.csv
        metrics/validation.json
        metrics/test.json
        predictions/test_predictions.csv
        checkpoints/best.pt
        checkpoints/last.pt
        figures/
      summaries/
        experiment_index.csv
        aggregate_metrics.csv
        findings.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

# Only allow characters that are safe on POSIX + Windows filesystems.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _check_safe(name: str, field: str) -> None:
    if not name or not _SAFE_ID_RE.match(name):
        raise ValueError(
            f"{field}={name!r} must match {_SAFE_ID_RE.pattern} "
            "(no path separators, no whitespace, no unicode)."
        )


@dataclass(frozen=True)
class RunLayout:
    """All paths for a single ``results/<run-id>/`` root."""

    root: Path

    @property
    def run_manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def resolved_config(self) -> Path:
        return self.root / "resolved_config.yaml"

    @property
    def environment(self) -> Path:
        return self.root / "environment.txt"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def orchestration_log(self) -> Path:
        return self.root / "logs" / "orchestration.log"

    @property
    def data_audit(self) -> Path:
        return self.root / "data_audit"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def summaries(self) -> Path:
        return self.root / "summaries"

    def experiment_dir(
        self, suite: str, split_id: str, model: str, seed: int
    ) -> Path:
        _check_safe(suite, "suite")
        _check_safe(split_id, "split_id")
        _check_safe(model, "model")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        return (
            self.experiments
            / suite
            / split_id
            / model
            / f"seed_{seed:d}"
        )

    def ensure(self) -> None:
        """Create the top-level scaffolding directories (idempotent)."""
        for d in (
            self.root,
            self.root / "logs",
            self.data_audit,
            self.splits,
            self.experiments,
            self.summaries,
        ):
            d.mkdir(parents=True, exist_ok=True)


def build_run_layout(run_id: str, root: Path | None = None) -> RunLayout:
    """Return the layout for ``<root>/<run_id>/``. Does not create dirs."""
    _check_safe(run_id, "run_id")
    root = Path(root) if root is not None else DEFAULT_RESULTS_ROOT
    return RunLayout(root=root / run_id)


def experiment_layout_files(exp_dir: Path) -> dict:
    """Return the canonical file paths inside one experiment directory."""
    return {
        "config": exp_dir / "config.yaml",
        "status": exp_dir / "status.json",
        "console_log": exp_dir / "logs" / "console.log",
        "events_log": exp_dir / "logs" / "events.jsonl",
        "epochs_csv": exp_dir / "metrics" / "epochs.csv",
        "validation_json": exp_dir / "metrics" / "validation.json",
        "test_json": exp_dir / "metrics" / "test.json",
        "predictions_csv": exp_dir / "predictions" / "test_predictions.csv",
        "checkpoint_best": exp_dir / "checkpoints" / "best.pt",
        "checkpoint_last": exp_dir / "checkpoints" / "last.pt",
        "figures_dir": exp_dir / "figures",
    }


def ensure_experiment_dir(exp_dir: Path) -> dict:
    """Create the sub-tree under ``exp_dir`` and return the file map."""
    files = experiment_layout_files(exp_dir)
    for k in ("console_log", "events_log", "epochs_csv", "predictions_csv",
              "checkpoint_best"):
        files[k].parent.mkdir(parents=True, exist_ok=True)
    files["figures_dir"].mkdir(parents=True, exist_ok=True)
    return files
