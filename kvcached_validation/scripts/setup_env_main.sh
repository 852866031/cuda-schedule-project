#!/usr/bin/env bash
# Second environment: identical to .venv except kvcached is built from the upstream main branch
# (commit recorded below) instead of the 0.1.6 release. Needed because 0.1.6 has no page pinning
# for vLLM's native OffloadingConnector; main (PR #598, 2026-10-27) adds it for vLLM 0.29/0.30.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
VENV="$HERE/.venv-main"
SRC=${KVCACHED_SRC:-/tmp/claude-1000/-home-jiaxuan-Documents-Projects-kvcache-vram-size/a2c0ccfd-5678-4cc9-a219-090f04cd538c/scratchpad/src/kvcached}
uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
uv pip install "vllm==0.30.0" setuptools wheel aiohttp numpy pandas matplotlib nvidia-ml-py requests
uv pip install "$SRC" --no-build-isolation
(cd "$SRC" && echo "kvcached source commit: $(git rev-parse HEAD) ($(git log -1 --format=%cd --date=short))")
python -c "import importlib.metadata as m; print('vllm', m.version('vllm'), 'kvcached', m.version('kvcached'))"
