#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${HOME}/.envs/.env/bin/activate"
cd "${PROJECT_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-codex}}"
mkdir -p "${MPLCONFIGDIR}"
python src/analyze_alignment_ablation.py "$@"
