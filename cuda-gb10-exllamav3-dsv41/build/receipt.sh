#!/bin/bash
# Records what was built, for humans and for the Tier-1 gate. Runs in the build
# stage (devel image, where cuobjdump exists) so the device-code inventory is
# captured into the shipped image as /app/CUDA_ARCHS.
set -euo pipefail

say() { echo "[exl3-receipt] $*"; }

EXL3_TREE="${EXL3_TREE:-$(ls -d /opt/venv/lib/python*/site-packages/exllamav3 2>/dev/null | head -1)}"
[ -n "${EXL3_TREE}" ] || { echo "[exl3-receipt] FATAL: exllamav3 package not found"; exit 1; }

# Keep this command identical to entrypoint.sh; the stamp is only meaningful if
# both sides hash the same set of files the same way.
tree_digest() {
    find "${EXL3_TREE}" -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}

mkdir -p /opt/llmkube
{
    echo "image: llmkube-exllamav3-cuda-gb10-dsv41-exl3"
    echo "exllamav3: $(python3 -c 'from exllamav3.version import __version__; print(__version__)')"
    echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "torch cuda: $(python3 -c 'import torch; print(torch.version.cuda)')"
    cat /opt/llmkube/patches/UPSTREAM_COMMITS.txt
    echo "build patch sha256: $(sha256sum /opt/llmkube/patches/0001-aarch64-build.patch | cut -d' ' -f1)"
    echo "device code: $(tr '\n' ' ' < /app/CUDA_ARCHS)"
} | tee /opt/llmkube/receipt.txt

tree_digest > /opt/llmkube/exllamav3-tree.sha256
say "exllamav3 tree sha256: $(cat /opt/llmkube/exllamav3-tree.sha256)"
