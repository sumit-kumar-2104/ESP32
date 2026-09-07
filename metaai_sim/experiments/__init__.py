"""Server-run evaluation harness.

This package provides a leakage-resistant, resumable, provenance-tracked
orchestration layer around the existing training scripts (train_raw_csi.py,
train_ab_dfs.py, b5_isolation.py, ...). It never fabricates metrics: every
number written into ``results/<run-id>/`` comes from an actual sub-process
or an in-process training call that touched the real dataset. Historical
numbers documented in the README are NOT copied here.

See docs/server_experiments.md for the operator contract.
"""

from experiments import (  # noqa: F401  (re-exports for callers/tests)
    manifest,
    metrics,
    paths,
    provenance,
    splits,
    status,
)

__all__ = ["manifest", "metrics", "paths", "provenance", "splits", "status"]
