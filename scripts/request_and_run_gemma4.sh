#!/usr/bin/env bash
#SBATCH --job-name=gemma4_tom_baseline
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-gemma4-%j.out

set -euo pipefail

# This wrapper supplies Gemma-specific defaults and delegates server lifecycle
# and evaluation to the shared, tested SLURM workflow.
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-google/gemma-4-26B-A4B-it}"
export RESULT_MODEL_NAME="${RESULT_MODEL_NAME:-Gemma-4-26B-A4B-it}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-gemma4}"
export VLLM_TEXT_ONLY="${VLLM_TEXT_ONLY:-1}"
export THINKING_TEMPERATURE="${THINKING_TEMPERATURE:-1.0}"
export THINKING_TOP_P="${THINKING_TOP_P:-0.95}"
export THINKING_TOP_K="${THINKING_TOP_K:-64}"
export THINKING_MIN_P="${THINKING_MIN_P:-0.0}"
export THINKING_PRESENCE_PENALTY="${THINKING_PRESENCE_PENALTY:-0.0}"
export THINKING_REPETITION_PENALTY="${THINKING_REPETITION_PENALTY:-1.0}"

exec "${PROJECT_ROOT}/scripts/request_and_run.sh"
