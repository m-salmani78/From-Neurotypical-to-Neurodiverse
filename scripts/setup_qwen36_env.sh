#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT_ACTIVATE="${HOME}/.envs/.env/bin/activate"
VLLM_VERSION="0.23.0"
# vLLM 0.23.0 publishes CUDA 12.9 (not 12.8) release wheels. CUDA 12.x
# minor-version compatibility allows this wheel on the cluster's 570 driver.
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

if [[ ! -f "${ENVIRONMENT_ACTIVATE}" ]]; then
    echo "Environment activation script not found: ${ENVIRONMENT_ACTIVATE}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENVIRONMENT_ACTIVATE}"

python -m pip install --upgrade \
    "${VLLM_WHEEL}" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    "openai>=1.99,<3" \
    "matplotlib>=3.7"

VERSIONS_DIR="${PROJECT_ROOT}/results/setup"
mkdir -p "${VERSIONS_DIR}"

python - <<'PY' | tee "${VERSIONS_DIR}/qwen36_environment_versions.txt"
import importlib.metadata
import platform

print(f"python={platform.python_version()}")
for package in ("vllm", "torch", "transformers", "openai", "matplotlib"):
    print(f"{package}={importlib.metadata.version(package)}")

import torch
import vllm
import flashinfer
import matplotlib
from openai import AsyncOpenAI

assert vllm.__version__ == "0.23.0", vllm.__version__
assert flashinfer.__version__ == importlib.metadata.version("flashinfer-cubin"), (
    flashinfer.__version__,
    importlib.metadata.version("flashinfer-cubin"),
)
assert AsyncOpenAI is not None
assert matplotlib is not None
print(f"torch_cuda_build={torch.version.cuda}")
print("imports=ok")
PY

echo "Qwen3.6 vLLM environment is ready."
