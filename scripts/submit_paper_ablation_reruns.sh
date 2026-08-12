#!/usr/bin/env bash
set -euo pipefail

# Submit only the four replacement cells required by the paper ablation.
# request_and_run.sh derives an isolated port from each SLURM job ID.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

ROLEPLAY_ONLY=1 \
RUN_ID=qwen-roleplay-paper-clean \
sbatch ./scripts/request_and_run.sh

ROLEPLAY_ONLY=1 ENABLE_THINKING=1 MAX_MODEL_LEN=16384 MAX_TOKENS=4096 \
RUN_ID=qwen-roleplay-thinking-paper-clean \
sbatch ./scripts/request_and_run.sh

ENABLE_THINKING=1 MAX_MODEL_LEN=16384 MAX_TOKENS=4096 \
RUN_ID=gemma4-thinking-paper-clean \
sbatch ./scripts/request_and_run_gemma4.sh

ROLEPLAY_ONLY=1 ENABLE_THINKING=1 MAX_MODEL_LEN=16384 MAX_TOKENS=4096 \
RUN_ID=gemma4-roleplay-thinking-paper-clean \
sbatch ./scripts/request_and_run_gemma4.sh

echo "Submitted four isolated ablation replacement jobs."
echo "After they finish: bash scripts/build_prompt_ablation.sh"
