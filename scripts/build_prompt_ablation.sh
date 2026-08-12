#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${HOME}/.envs/.env/bin/activate"
cd "${PROJECT_ROOT}"
python src/analyze_prompt_ablation.py "$@"
