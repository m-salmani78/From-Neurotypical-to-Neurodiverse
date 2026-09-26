#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT_ACTIVATE="${HOME}/.envs/.env/bin/activate"
VLLM_VERSION="0.23.0"
# The vLLM 0.23.0 release provides a CUDA-12.9 extension wheel, but no cu128
# asset. CUDA 12 minor releases are compatible with this cluster driver.
# Its CPython-3.12 torchaudio dependency is unusable here: the cu129 wheel
# links against libcudart.so.13 although this environment provides CUDA 12.x.
# Audio is optional for this text-only experiment, so setup removes torchaudio after
# installing vLLM. Transformers then does not attempt to import it at startup.
VLLM_CUDA_VARIANT="cu129"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_CUDA_VARIANT}-cp38-abi3-manylinux_2_28_x86_64.whl"

if [[ ! -f "${ENVIRONMENT_ACTIVATE}" ]]; then
    echo "Environment activation script not found: ${ENVIRONMENT_ACTIVATE}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENVIRONMENT_ACTIVATE}"

# Do not reinstall the multi-gigabyte Torch stack. The existing CUDA-12.8
# Torch and torchvision builds share the required CUDA-12 ABI with vLLM's
# cu129 extension.
python -m pip install --upgrade \
    "${VLLM_WHEEL}" \
    --extra-index-url "https://download.pytorch.org/whl/${VLLM_CUDA_VARIANT}" \
    "openai>=1.99,<3" \
    "sentencepiece>=0.2,<0.3" \
    "matplotlib>=3.7"

# This is intentionally absent rather than replaced with a mismatched CUDA
# build. It is only required for audio-capable models, which this workflow does
# not serve. Keep the uninstall tolerant of a fresh environment.
python -m pip uninstall --yes torchaudio || true
# Some pip releases leave a torchaudio *.dist-info directory behind after the
# uninstall. Transformers uses package metadata for optional-import detection,
# so remove only those stale metadata directories as well.
python - <<'PY'
import importlib.metadata
import shutil

for distribution in importlib.metadata.distributions():
    if (distribution.metadata.get("Name") or "").lower() == "torchaudio":
        print(f"Removing stale torchaudio metadata: {distribution._path}")
        shutil.rmtree(distribution._path)
PY

VERSIONS_DIR="${PROJECT_ROOT}/results/setup"
mkdir -p "${VERSIONS_DIR}"

python - <<'PY' | tee "${VERSIONS_DIR}/qwen36_environment_versions.txt"
import importlib.metadata
import platform

print(f"python={platform.python_version()}")
for package in ("torch", "transformers", "openai", "matplotlib", "sentencepiece", "flashinfer-python", "flashinfer-cubin"):
    print(f"{package}={importlib.metadata.version(package)}")

import torch
import torchvision
import vllm
import flashinfer
import matplotlib
import sentencepiece
from openai import AsyncOpenAI

vllm_distributions = sorted(
    distribution.version
    for distribution in importlib.metadata.distributions()
    if (distribution.metadata.get("Name") or "").lower() == "vllm"
)
print(f"vllm_distributions={vllm_distributions}")
print(f"vllm_runtime={vllm.__version__}")
assert vllm.__version__ == "0.23.0", vllm.__version__
assert len(vllm_distributions) == 1, vllm_distributions
assert vllm_distributions[0].split("+")[0] == "0.23.0", vllm_distributions
assert torch.version.cuda == "12.8", torch.version.cuda
assert "+cu128" in torch.__version__, torch.__version__
assert "+cu128" in torchvision.__version__, torchvision.__version__
try:
    importlib.metadata.version("torchaudio")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("torchaudio must be absent for this text-only environment")
assert AsyncOpenAI is not None
assert matplotlib is not None
assert flashinfer is not None
assert sentencepiece is not None
print(f"torch_cuda_build={torch.version.cuda}")
print("torchaudio=absent (text-only compatibility mode)")
print(f"torchvision={torchvision.__version__}")
print("imports=ok")
PY

echo "Qwen3.6 vLLM environment is ready."
