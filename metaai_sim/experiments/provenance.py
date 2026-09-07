"""Run manifest / provenance capture.

Records enough to reproduce a run: git commit + dirty-tree flag, exact
launch command, resolved config, dependency versions, selected device,
dataset/feature-cache/split identities, seeds, models, and per-status
experiment counts. Deliberately conservative about ``environment.txt``:
we NEVER dump ``os.environ`` (which would leak SSH agent sockets, tokens,
proxy credentials, kerberos ticket paths, ...). Only known-safe keys are
recorded, each labelled with its source.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Environment variables that are (a) documented inputs for this repo and
# (b) safe to record verbatim. Anything else stays out.
SAFE_ENV_KEYS = (
    "METAAI_DATA_DIR",
    "METAAI_RAW_CSI_DIR",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git(cwd: Path, *args: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=str(cwd), stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""
    return out.decode("utf-8", errors="replace").strip()


def git_info(repo: Path) -> dict[str, Any]:
    """Return commit hash, short hash, dirty-tree flag and branch."""
    repo = Path(repo)
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(repo, "status", "--porcelain")
    dirty_files = [ln for ln in status.splitlines() if ln.strip()]
    return {
        "commit": commit,
        "short_commit": commit[:12] if commit else "",
        "branch": branch,
        "dirty": bool(dirty_files),
        "dirty_file_count": len(dirty_files),
    }


def device_info() -> dict[str, Any]:
    """Return selected device + driver info without importing GPU libs eagerly."""
    info: dict[str, Any] = {"selected": "cpu", "cuda_available": False}
    try:
        import torch
    except ImportError:
        return info
    info["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["selected"] = "cuda"
        info["cuda_device_count"] = torch.cuda.device_count()
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover — device-specific
            info["cuda_device_name"] = "unknown"
    return info


def dependency_versions() -> dict[str, str]:
    """Return versions of the direct dependencies we actually import."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "torch", "sklearn", "scipy", "yaml"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[mod] = "not-installed"
    return versions


def safe_env_snapshot(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return only the explicitly documented env vars."""
    import os as _os
    env = env if env is not None else dict(_os.environ)
    return {k: env[k] for k in SAFE_ENV_KEYS if k in env}


@dataclass
class RunManifest:
    run_id: str
    profile: str
    launch_cmd: list[str]
    started_at: str = field(default_factory=_now_iso)
    ended_at: str | None = None
    seeds: list[int] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    suites: list[str] = field(default_factory=list)
    dataset_dir: str | None = None
    raw_csi_dir: str | None = None
    feature_cache_ids: list[str] = field(default_factory=list)
    split_ids: list[str] = field(default_factory=list)
    git: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: {
        "pending": 0, "running": 0, "completed": 0,
        "failed": 0, "unavailable": 0,
    })

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def write_environment_txt(path: Path) -> None:
    """Write a conservative ``environment.txt`` (versions + safe env only)."""
    parts: list[str] = []
    parts.append("# environment.txt — recorded by experiments.provenance")
    parts.append(f"# generated_at: {_now_iso()}")
    parts.append("")
    parts.append("[versions]")
    for k, v in dependency_versions().items():
        parts.append(f"{k}={v}")
    parts.append("")
    parts.append("[device]")
    for k, v in device_info().items():
        parts.append(f"{k}={v}")
    parts.append("")
    parts.append("[env]  # only allow-listed vars — see SAFE_ENV_KEYS")
    for k, v in safe_env_snapshot().items():
        parts.append(f"{k}={v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
