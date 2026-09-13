#!/bin/bash
#
# Optional launcher. LLMKube's vLLM backend sets the container command to ["vllm","serve"], which
# overrides this ENTRYPOINT, so on the operator path this does not run. The image is built to be
# correct without it: every overlay is applied and tested at build. This is a second line of defence
# for `docker run` and for a spec that opts in with command: ["/usr/local/bin/dsv41-entrypoint.sh","vllm","serve"].
set -euo pipefail
say() { echo "[dsv41-exl3] $*"; }
die() { echo "[dsv41-exl3] FATAL: $*" >&2; exit 1; }
VLLM_SITE="${VLLM_SITE:-/usr/local/lib/python3.12/dist-packages/vllm}"
STAMP=/opt/llmkube/vllm-tree.sha256
if [ "${DSV41_VERIFY_TREE:-0}" = "1" ]; then
    [ -f "$STAMP" ] || die "no build stamp at $STAMP; this is not a properly built image"
    want="$(cat "$STAMP")"
    got="$(find "$VLLM_SITE" -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
    [ "$want" = "$got" ] || die "vLLM tree digest mismatch: built=${want} now=${got}; something modified site-packages after build"
    say "vLLM tree digest verified: ${got}"
fi
[ "$#" -gt 0 ] || die "no command given; the InferenceService supplies the vllm serve invocation"
say "exec: $*"
exec "$@"
