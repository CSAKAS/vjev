#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CACHE_ROOT=/tmp/alansong-rgb-benchmark/vllm
export TRITON_CACHE_DIR=/tmp/alansong-rgb-benchmark/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/alansong-rgb-benchmark/torchinductor
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -u order.py "$@"
