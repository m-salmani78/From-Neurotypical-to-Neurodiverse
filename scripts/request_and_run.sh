#!/usr/bin/env bash
#SBATCH --job-name=qwen36_tom_baseline
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

set -euo pipefail

# sbatch runs a spool copy of this file under /var/slurm. SLURM_SUBMIT_DIR is
# the repository directory from which `sbatch ./scripts/request_and_run.sh`
# was invoked.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.6-27B}"
RESULT_MODEL_NAME="${RESULT_MODEL_NAME:-Qwen3.6-27B}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-}"
VLLM_TEXT_ONLY="${VLLM_TEXT_ONLY:-0}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
if [[ -z "${SERVER_PORT+x}" ]]; then
    if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
        SERVER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
    else
        SERVER_PORT=8000
    fi
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
ROLEPLAY_ONLY="${ROLEPLAY_ONLY:-0}"
if [[ "${ENABLE_THINKING}" == "1" ]]; then
    DEFAULT_MAX_MODEL_LEN=16384
    DEFAULT_MAX_TOKENS=4096
    DEFAULT_TEMPERATURE="${THINKING_TEMPERATURE:-1.0}"
    DEFAULT_TOP_P="${THINKING_TOP_P:-0.95}"
    DEFAULT_TOP_K="${THINKING_TOP_K:-20}"
    DEFAULT_MIN_P="${THINKING_MIN_P:-0.0}"
    DEFAULT_PRESENCE_PENALTY="${THINKING_PRESENCE_PENALTY:-0.0}"
    DEFAULT_REPETITION_PENALTY="${THINKING_REPETITION_PENALTY:-1.0}"
elif [[ "${ENABLE_THINKING}" == "0" ]]; then
    DEFAULT_MAX_MODEL_LEN=8192
    DEFAULT_MAX_TOKENS=512
    DEFAULT_TEMPERATURE=0.2
    DEFAULT_TOP_P=0.5
    DEFAULT_TOP_K=""
    DEFAULT_MIN_P=""
    DEFAULT_PRESENCE_PENALTY=""
    DEFAULT_REPETITION_PENALTY=""
else
    echo "ENABLE_THINKING must be 0 or 1 (received: ${ENABLE_THINKING})." >&2
    exit 2
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
export MAX_MODEL_LEN
TEMPERATURE="${TEMPERATURE:-${DEFAULT_TEMPERATURE}}"
TOP_P="${TOP_P:-${DEFAULT_TOP_P}}"
TOP_K="${TOP_K:-${DEFAULT_TOP_K}}"
MIN_P="${MIN_P:-${DEFAULT_MIN_P}}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-${DEFAULT_PRESENCE_PENALTY}}"
REPETITION_PENALTY="${REPETITION_PENALTY:-${DEFAULT_REPETITION_PENALTY}}"
if [[ "${ROLEPLAY_ONLY}" != "0" && "${ROLEPLAY_ONLY}" != "1" ]]; then
    echo "ROLEPLAY_ONLY must be 0 or 1 (received: ${ROLEPLAY_ONLY})." >&2
    exit 2
fi
if [[ "${VLLM_TEXT_ONLY}" != "0" && "${VLLM_TEXT_ONLY}" != "1" ]]; then
    echo "VLLM_TEXT_ONLY must be 0 or 1 (received: ${VLLM_TEXT_ONLY})." >&2
    exit 2
fi
RESULT_VARIANT="${RESULT_MODEL_NAME}"
if [[ "${ROLEPLAY_ONLY}" == "1" ]]; then
    RESULT_VARIANT+="-roleplay-only"
fi
if [[ "${ENABLE_THINKING}" == "1" ]]; then
    RESULT_VARIANT+="-thinking"
fi
MAX_TOKENS="${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"
CONCURRENCY="${CONCURRENCY:-16}"
LIMIT="${LIMIT:-0}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
RUN_ID="${RUN_ID:-slurm-${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/baseline/${RESULT_VARIANT}/${RUN_ID}}"

source ~/.envs/.env/bin/activate
if [[ ! -f "${PROJECT_ROOT}/src/run_model_baseline.py" || ! -d "${PROJECT_ROOT}/Data" ]]; then
    echo "PROJECT_ROOT is not this repository: ${PROJECT_ROOT}" >&2
    echo "Submit from the repository root with: sbatch ./scripts/request_and_run.sh" >&2
    exit 1
fi
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

# FlashInfer's sampler tries to JIT-compile on A100 when a matching prebuilt
# sampling kernel is unavailable. Compute nodes do not provide nvcc, so use
# vLLM's native PyTorch sampler instead. This does not disable model attention.
export VLLM_USE_FLASHINFER_SAMPLER=0

python - <<'PY'
import importlib.metadata

for package in ("vllm", "openai", "flashinfer-python", "flashinfer-cubin"):
    try:
        print(f"{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            f"{package} is not installed. Run: bash scripts/setup_qwen36_env.sh"
        ) from exc
PY
nvidia-smi

SERVER_LOG="${OUTPUT_DIR}/server.log"
SERVER_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

VLLM_ARGS=(
    serve "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${SERVER_HOST}"
    --port "${SERVER_PORT}"
    --dtype bfloat16
    --tensor-parallel-size 1
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)
if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
    VLLM_ARGS+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
fi
if [[ "${VLLM_TEXT_ONLY}" == "1" ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt '{"image":0,"audio":0}')
fi

if curl --silent --fail --max-time 2 \
    "http://${SERVER_HOST}:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Port ${SERVER_HOST}:${SERVER_PORT} already has a vLLM-compatible server." >&2
    echo "Refusing to connect this experiment to a server owned by another job." >&2
    exit 1
fi

vllm "${VLLM_ARGS[@]}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Started vLLM as PID ${SERVER_PID}; waiting for readiness."
SECONDS_WAITED=0
MODELS_RESPONSE=""
while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM exited before becoming ready." >&2
        tail -200 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if MODELS_RESPONSE="$(
        curl --silent --fail --max-time 5 \
            "http://${SERVER_HOST}:${SERVER_PORT}/v1/models"
    )"; then
        break
    fi
    if (( SECONDS_WAITED >= SERVER_READY_TIMEOUT )); then
        echo "Timed out after ${SERVER_READY_TIMEOUT}s waiting for vLLM." >&2
        tail -200 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 10
    SECONDS_WAITED=$((SECONDS_WAITED + 10))
done

export MODELS_RESPONSE
python - "${SERVED_MODEL_NAME}" <<'PY'
import json
import os
import sys

expected = sys.argv[1]
payload = json.loads(os.environ["MODELS_RESPONSE"])
served = {entry.get("id") for entry in payload.get("data", [])}
if expected not in served:
    raise SystemExit(
        f"vLLM readiness model mismatch: expected {expected!r}, served {sorted(served)!r}"
    )
print(f"Verified served model: {expected}")
PY
unset MODELS_RESPONSE

echo "vLLM is ready; starting baseline evaluation."
RUNNER_ARGS=(
    --base-url "http://${SERVER_HOST}:${SERVER_PORT}/v1"
    --model "${SERVED_MODEL_NAME}"
    --data-dir "${PROJECT_ROOT}/Data"
    --output-dir "${OUTPUT_DIR}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${CONCURRENCY}"
    --resume
)
if (( LIMIT > 0 )); then
    RUNNER_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${ENABLE_THINKING}" == "1" ]]; then
    RUNNER_ARGS+=(--enable-thinking)
fi
if [[ "${ROLEPLAY_ONLY}" == "1" ]]; then
    RUNNER_ARGS+=(--roleplay-only)
fi
if [[ -n "${TOP_K}" ]]; then
    RUNNER_ARGS+=(--top-k "${TOP_K}")
fi
if [[ -n "${MIN_P}" ]]; then
    RUNNER_ARGS+=(--min-p "${MIN_P}")
fi
if [[ -n "${PRESENCE_PENALTY}" ]]; then
    RUNNER_ARGS+=(--presence-penalty "${PRESENCE_PENALTY}")
fi
if [[ -n "${REPETITION_PENALTY}" ]]; then
    RUNNER_ARGS+=(--repetition-penalty "${REPETITION_PENALTY}")
fi

python src/run_model_baseline.py "${RUNNER_ARGS[@]}"
echo "Baseline evaluation completed: ${OUTPUT_DIR}"
