#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Records what was built, for humans and for the Tier-1 gate.
set -euo pipefail
D=/usr/local/lib/python3.12/dist-packages
mkdir -p /opt/llmkube
{
  echo "image: llmkube-vllm-cuda-gb10-dsv41-kk"
  echo "vllm: $(python3 -c 'import vllm; print(vllm.__version__)')"
  echo "vllm install: $(cat /opt/llmkube/vllm-install.txt)"
  echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"
  echo "flashinfer: $(python3 -c 'import flashinfer; print(flashinfer.__version__)')"
  echo "b12x: $(python3 -c 'import importlib.metadata as m; print(m.version("b12x"))')"
  echo "nvidia-cutlass-dsl: $(python3 -c 'import importlib.metadata as m; print(m.version("nvidia-cutlass-dsl"))')"
  echo "nccl: $(python3 -c 'import importlib.metadata as m; print(m.version("nvidia-nccl-cu13"))') (library rebuilt from source, below)"
  sed 's/^/  /' /opt/llmkube/nccl-build.txt
  cat /opt/llmkube/patches/UPSTREAM_COMMITS.txt
  echo "b12x compile env:"
  sed 's/^/  /' /opt/llmkube/b12x-compile-env.txt
} > /opt/llmkube/receipt.txt
# Full resolution, since the fork's requirements leave most runtime deps as ranges.
pip list --format=freeze > /opt/llmkube/pip-freeze.txt
find "$D/vllm" "$D/b12x" -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1 > /opt/llmkube/python-tree.sha256
cat /opt/llmkube/receipt.txt
echo "vllm+b12x tree sha256: $(cat /opt/llmkube/python-tree.sha256)"
