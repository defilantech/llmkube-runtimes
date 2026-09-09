#!/bin/bash
#
# Optional launcher for the GLM-5.3-Flash EXL3 runtime.
#
# WHEN THIS RUNS, WHICH IS NOT ALWAYS. READ THIS FIRST.
#
# A Kubernetes container `command` OVERRIDES the image ENTRYPOINT, and the
# LLMKube vLLM backend sets Command to ["vllm","serve"] for every vLLM
# InferenceService (internal/controller/runtime_vllm.go, BuildCommand). So on
# the normal operator path this script does NOT execute.
#
# The image is therefore built to be correct without it. Every overlay patch is
# applied at BUILD, and upstream's own test suite runs at build against the
# patched tree, so what ships is already patched. Nothing here is required for
# correctness; this is a second line of defence for `docker run` and for a spec
# that opts in:
#
#     spec:
#       command: ["/usr/local/bin/glm53-entrypoint.sh", "vllm", "serve"]
#
# WHY IT VERIFIES RATHER THAN RE-APPLIES
#
# An earlier version of this file re-ran the patch scripts at start. That was
# right when the overlay was bind-mounted over a stock base, and wrong here: the
# patches are baked into the image layers, so re-running them can only either be
# a no-op or, if the tree has been mounted over, half-apply against files the
# patches were never verified against. Comparing a digest recorded at build is
# strictly more informative and cannot itself corrupt anything.

set -euo pipefail

say() { echo "[glm53-exl3] $*"; }
die() { echo "[glm53-exl3] FATAL: $*" >&2; exit 1; }

VLLM_SITE="${VLLM_SITE:-/usr/local/lib/python3.12/dist-packages/vllm}"
STAMP=/opt/llmkube/vllm-tree.sha256

# Abliteration is not shipped in this image. Upstream's ABLIT=1 path removes the
# model's refusal direction; neither the donor artifacts nor the runtime hook
# are built in, so the switch has nothing to load. Refuse rather than start and
# let an operator believe a setting took effect that did not.
if [ "${ABLIT:-0}" != "0" ]; then
    die "ABLIT=${ABLIT} requested, but this image ships no abliteration artifacts by design."
fi

# Off by default because hashing the tree costs a few seconds on every pod
# start, and the build already proved the tree. Turn on where a mutating
# sidecar or a hostPath mount over site-packages is a real possibility.
if [ "${GLM53_VERIFY_TREE:-0}" = "1" ]; then
    [ -f "$STAMP" ] || die "no build stamp at $STAMP; this is not a properly built image"
    want="$(cat "$STAMP")"
    got="$(find "$VLLM_SITE" -name '*.py' -type f -print0 \
             | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
    if [ "$want" != "$got" ]; then
        die "vLLM tree digest mismatch. built=${want} now=${got}. Something modified site-packages after build; refusing to serve."
    fi
    say "vLLM tree digest verified: ${got}"
fi

[ "$#" -gt 0 ] || die "no command given; the InferenceService must supply the vllm serve invocation"

say "exec: $*"
exec "$@"
