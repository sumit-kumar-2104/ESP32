#!/usr/bin/env bash
# Thin wrapper around python -m experiments.runner so the exact server
# command contract in docs/server_experiments.md stays stable.
#
# Usage:
#   bash scripts/run_all.sh --profile smoke --device auto
#   bash scripts/run_all.sh --profile core  --device auto
#   bash scripts/run_all.sh --profile full  --device auto
#   bash scripts/run_all.sh --resume <run-id> --device auto
#
# Deliberately does NOT install packages, does NOT download data, does NOT
# spawn background jobs. If METAAI_DATA_DIR / METAAI_RAW_CSI_DIR are
# required by the chosen profile, the runner will fail loudly with a
# specific message; we never guess a fallback dataset here.

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"

cd "${REPO_ROOT}"
exec python -m experiments.runner "$@"
