#!/usr/bin/env bash
#SBATCH --job-name=llama3_alignment
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-3%4
#SBATCH --output=slurm-alignment-%A_%a.out

set -euo pipefail

# One array task serves one validated checkpoint and evaluates both prompt modes.
# SimPO is excluded because its tokenizer cannot be instantiated in this cluster
# environment. Submit from the repository root with: sbatch ./scripts/run_alignment_ablation.sh
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.5}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-512}"
CONCURRENCY="${CONCURRENCY:-16}"
LIMIT="${LIMIT:-0}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
RUN_ID="${RUN_ID:-seed-${SEED}}"

MODELS=(
    "princeton-nlp/Llama-3-Base-8B-SFT"
    "princeton-nlp/Llama-3-Base-8B-SFT-DPO"
    "princeton-nlp/Llama-3-Base-8B-SFT-ORPO"
    "princeton-nlp/Llama-3-Base-8B-SFT-KTO"
)
METHODS=("SFT" "DPO" "ORPO" "KTO")

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-}"
if [[ ! "${ARRAY_INDEX}" =~ ^[0-3]$ ]]; then
    echo "This script requires SLURM_ARRAY_TASK_ID in 0..3." >&2
    echo "Submit it with: sbatch ./scripts/run_alignment_ablation.sh" >&2
    exit 2
fi
MODEL="${MODELS[ARRAY_INDEX]}"
ALIGNMENT_METHOD="${METHODS[ARRAY_INDEX]}"
SERVED_MODEL_NAME="${MODEL}"

if [[ -z "${SERVER_PORT+x}" ]]; then
    ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
    SERVER_PORT="$((20000 + (ARRAY_JOB_ID * 10 + ARRAY_INDEX) % 20000))"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/alignment_ablation/${ALIGNMENT_METHOD}/${RUN_ID}}"
FULL_OUTPUT="${OUTPUT_ROOT}/full_profile"
ROLE_OUTPUT="${OUTPUT_ROOT}/roleplay_only"
TOKENIZER_PREFLIGHT="${OUTPUT_ROOT}/tokenizer_preflight.json"
SERVER_LOG="${OUTPUT_ROOT}/server-${SLURM_JOB_ID:-manual}.log"
JOB_MANIFEST="${OUTPUT_ROOT}/job-${SLURM_JOB_ID:-manual}.json"

source "${HOME}/.envs/.env/bin/activate"
if [[ ! -f "${PROJECT_ROOT}/src/run_model_baseline.py" || ! -d "${PROJECT_ROOT}/Data" ]]; then
    echo "PROJECT_ROOT is not this repository: ${PROJECT_ROOT}" >&2
    exit 1
fi
cd "${PROJECT_ROOT}"
mkdir -p "${FULL_OUTPUT}" "${ROLE_OUTPUT}"

python - <<'PY'
import importlib.metadata
import torch
import torchvision
import vllm
try:
    import sentencepiece
except ImportError as exc:
    raise SystemExit(
        "sentencepiece is required for the slow Llama tokenizer. "
        "Run: bash scripts/setup_qwen36_env.sh"
    ) from exc

vllm_distributions = sorted(
    distribution.version
    for distribution in importlib.metadata.distributions()
    if (distribution.metadata.get("Name") or "").lower() == "vllm"
)
vllm_runtime = vllm.__version__
openai_version = importlib.metadata.version("openai")
sentencepiece_version = importlib.metadata.version("sentencepiece")
if (
    vllm_runtime != "0.23.0"
    or len(vllm_distributions) != 1
    or vllm_distributions[0].split("+")[0] != "0.23.0"
    or int(openai_version.split(".")[0]) >= 3
    or torch.version.cuda != "12.8"
    or "+cu128" not in torch.__version__
    or "+cu128" not in torchvision.__version__
):
    raise SystemExit(
        "Environment version mismatch: expected one vllm 0.23.0 distribution, "
        "vllm runtime 0.23.0, openai <3, and CUDA-12.8 torch/torchvision; "
        f"found distributions {vllm_distributions}, runtime {vllm_runtime}, "
        f"openai {openai_version}, torch {torch.__version__} (CUDA "
        f"{torch.version.cuda}), and torchvision {torchvision.__version__}. "
        "Run: bash scripts/setup_qwen36_env.sh"
    )
try:
    importlib.metadata.version("torchaudio")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit(
        "torchaudio is installed, but its CUDA binary is incompatible with this "
        "text-only vLLM environment. Run: bash scripts/setup_qwen36_env.sh"
    )
print(
    f"Validated inference environment: vllm={vllm_runtime} "
    f"({vllm_distributions[0]}), openai={openai_version}, "
    f"sentencepiece={sentencepiece_version}"
)
PY

# Resolve and pin the immutable Hub revision once. Authentication, if needed,
# is read internally and is never echoed or written to the result manifests.
MODEL_REVISION="${MODEL_REVISION:-$(python - "${MODEL}" <<'PY'
import os
import sys
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN") or None
print(HfApi(token=token).model_info(sys.argv[1]).sha)
PY
)}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-${MODEL_REVISION}}"
export MODEL_REVISION TOKENIZER_REVISION TOKENIZER_PREFLIGHT ALIGNMENT_METHOD MAX_MODEL_LEN
export TEMPERATURE TOP_P SEED MAX_TOKENS

# A resumed directory must never mix responses from two checkpoint revisions.
python - "${MODEL}" "${MODEL_REVISION}" "${FULL_OUTPUT}/manifest.json" "${ROLE_OUTPUT}/manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

model, revision, *manifest_paths = sys.argv[1:]
expected_modes = ("full_profile", "roleplay_only")
expected_decoding = {
    "temperature": float(os.environ["TEMPERATURE"]),
    "top_p": float(os.environ["TOP_P"]),
    "seed": int(os.environ["SEED"]),
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "enable_thinking": False,
}
for raw_path, prompt_mode in zip(manifest_paths, expected_modes):
    path = Path(raw_path)
    if not path.exists():
        continue
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("model") != model or previous.get("model_revision") != revision:
        raise SystemExit(
            f"Refusing mixed-revision resume in {path.parent}: existing "
            f"{previous.get('model')}@{previous.get('model_revision')}, requested "
            f"{model}@{revision}. Use a new RUN_ID."
        )
    if (
        previous.get("alignment_method") != os.environ["ALIGNMENT_METHOD"]
        or previous.get("prompt_mode") != prompt_mode
        or previous.get("max_model_len") != int(os.environ["MAX_MODEL_LEN"])
        or any(previous.get("decoding", {}).get(key) != value for key, value in expected_decoding.items())
    ):
        raise SystemExit(
            f"Refusing mixed-configuration resume in {path.parent}. Use a new RUN_ID."
        )
PY

# The released Llama-3 checkpoints require exactly one BOS token. Validate the
# checkpoint's own chat template before vLLM starts and record its fingerprint.
python - "${MODEL}" "${TOKENIZER_REVISION}" "${TOKENIZER_PREFLIGHT}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer

model, revision, output = sys.argv[1:]
tokenizer = AutoTokenizer.from_pretrained(
    model,
    revision=revision,
    token=os.environ.get("HF_TOKEN") or None,
    # The Princeton Llama checkpoints expose only a slow SentencePiece
    # tokenizer under Transformers 5; auto/fast loading fails before serving.
    use_fast=False,
)
template = tokenizer.get_chat_template()
messages = [
    {"role": "system", "content": "Role: Neurotypical"},
    {"role": "user", "content": "Answer: [[1]]"},
]
ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
# Transformers 5 may return a BatchEncoding (with input_ids) rather than a
# plain token-id list when the tokenizer's chat template requests tensors.
if isinstance(ids, dict) or hasattr(ids, "data"):
    ids = ids["input_ids"]
if hasattr(ids, "tolist"):
    ids = ids.tolist()
if ids and isinstance(ids[0], list):
    ids = ids[0]
if not isinstance(ids, list):
    raise SystemExit(f"Tokenizer preflight returned unsupported token IDs: {type(ids)!r}")
bos_id = tokenizer.bos_token_id
bos_count = ids.count(bos_id) if bos_id is not None else 0
payload = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "model": model,
    "tokenizer_revision": revision,
    "tokenizer_class": type(tokenizer).__name__,
    "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    "bos_token_id": bos_id,
    "bos_count": bos_count,
    "token_count": len(ids),
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
if bos_count != 1:
    raise SystemExit(
        f"Tokenizer preflight failed: expected exactly one BOS token, found {bos_count}"
    )
print(f"Tokenizer preflight passed at revision {revision}: one BOS token")
PY

python - "${JOB_MANIFEST}" "${MODEL}" "${MODEL_REVISION}" "${ALIGNMENT_METHOD}" "${SERVER_PORT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "model": sys.argv[2],
    "model_revision": sys.argv[3],
    "alignment_method": sys.argv[4],
    "comparison_group": "SFT-only" if sys.argv[4] == "SFT" else "post-SFT preference optimization",
    "server_port": int(sys.argv[5]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "prompt_modes": ["full_profile", "roleplay_only"],
    "native_thinking": False,
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi

SERVER_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    if (( status != 0 )); then
        python - "${JOB_MANIFEST}" "${status}" <<'PY' || true
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
payload.update({"status": "failed", "completed_at": datetime.now(timezone.utc).isoformat(), "exit_status": int(sys.argv[2])})
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

if curl --silent --fail --max-time 2 \
    "http://${SERVER_HOST}:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Port ${SERVER_HOST}:${SERVER_PORT} is already occupied; refusing cross-job reuse." >&2
    exit 1
fi

vllm serve "${MODEL}" \
    --revision "${MODEL_REVISION}" \
    --tokenizer "${MODEL}" \
    --tokenizer-revision "${TOKENIZER_REVISION}" \
    --tokenizer-mode slow \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Serving ${MODEL}@${MODEL_REVISION} as PID ${SERVER_PID} on port ${SERVER_PORT}."
SECONDS_WAITED=0
MODELS_RESPONSE=""
while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM exited before readiness." >&2
        tail -200 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if MODELS_RESPONSE="$(curl --silent --fail --max-time 5 "http://${SERVER_HOST}:${SERVER_PORT}/v1/models")"; then
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
served = {entry.get("id") for entry in json.loads(os.environ["MODELS_RESPONSE"]).get("data", [])}
if expected not in served:
    raise SystemExit(f"served-model mismatch: expected {expected!r}, got {sorted(served)!r}")
print(f"Verified served model: {expected}")
PY
unset MODELS_RESPONSE

COMMON_ARGS=(
    --base-url "http://${SERVER_HOST}:${SERVER_PORT}/v1"
    --model "${SERVED_MODEL_NAME}"
    --data-dir "${PROJECT_ROOT}/Data"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${CONCURRENCY}"
    --resume
)
if (( LIMIT > 0 )); then
    COMMON_ARGS+=(--limit "${LIMIT}")
fi

python src/run_model_baseline.py "${COMMON_ARGS[@]}" --output-dir "${FULL_OUTPUT}"
python src/run_model_baseline.py "${COMMON_ARGS[@]}" --roleplay-only --output-dir "${ROLE_OUTPUT}"

python - "${JOB_MANIFEST}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "exit_status": 0})
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "Alignment ablation completed: ${OUTPUT_ROOT}"
