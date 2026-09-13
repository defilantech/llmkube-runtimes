#!/bin/bash
# Records what was built, for humans and for the Tier-1 gate (spec 5.6).
set -euo pipefail
D=/usr/local/lib/python3.12/dist-packages
mkdir -p /opt/llmkube
{
  echo "image: llmkube-vllm-cuda-gb10-dsv41-exl3"
  echo "vllm: $(python3 -c 'import vllm; print(vllm.__version__)')"
  echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"
  echo "flashinfer: $(python3 -c 'import flashinfer; print(flashinfer.__version__)')"
  echo "cuda_exl3: $(python3 -c 'import importlib.metadata as m; print(m.version("cuda-exl3"))')"
  cat /opt/llmkube/patches/UPSTREAM_COMMITS.txt
  echo "prewarmed: $(find "${FLASHINFER_WORKSPACE_BASE}" -name '*.so' -printf '%f ' 2>/dev/null)"
} > /opt/llmkube/receipt.txt
find "$D/vllm" -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1 > /opt/llmkube/vllm-tree.sha256
cat /opt/llmkube/receipt.txt
echo "vllm tree sha256: $(cat /opt/llmkube/vllm-tree.sha256)"
