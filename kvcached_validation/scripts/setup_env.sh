#!/usr/bin/env bash
# Build a fresh, self-contained environment for the kvcached validation.
# Versions chosen from the kvcached README (vLLM tested up to 0.30.0, torch >= 2.10)
# and the LMCache compatibility page (external MP connector needs vLLM >= 0.20;
# the PyPI 0.5.x wheel is the CUDA 13 build matching torch 2.13 / vLLM 0.30).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
VENV="$HERE/.venv"
VLLM_VERSION=${VLLM_VERSION:-0.30.0}
KVCACHED_VERSION=${KVCACHED_VERSION:-0.1.6}
LMCACHE_VERSION=${LMCACHE_VERSION:-0.5.6}

uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
uv pip install "vllm==${VLLM_VERSION}" "lmcache==${LMCACHE_VERSION}"
# kvcached is an sdist that compiles against the installed torch.
uv pip install setuptools wheel
uv pip install "kvcached==${KVCACHED_VERSION}" --no-build-isolation
# load generator / analysis helpers
uv pip install aiohttp numpy pandas matplotlib pynvml requests
python - <<'PY'
import importlib.metadata as m, torch
for p in ["vllm","kvcached","lmcache","torch","transformers","flashinfer-python","cuda-python","cupy-cuda13x"]:
    try: print(f"{p}=={m.version(p)}")
    except Exception as e: print(p, "MISSING", e)
print("torch.cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
