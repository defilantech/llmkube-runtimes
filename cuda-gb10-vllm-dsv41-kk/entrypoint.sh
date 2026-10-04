#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Optional launcher. LLMKube's vLLM backend sets the container command to ["vllm","serve"], which
# overrides this ENTRYPOINT, so on the operator path this does not run. The image is built to be
# correct without it: the pins gate and the in-image tests run at build. This is a second line of defence
# for `docker run` and for a spec that opts in with command: ["/usr/local/bin/dsv41-entrypoint.sh","vllm","serve"].
set -euo pipefail
say() { echo "[dsv41-kk] $*"; }
die() { echo "[dsv41-kk] FATAL: $*" >&2; exit 1; }
SITE="${SITE:-/usr/local/lib/python3.12/dist-packages}"
STAMP=/opt/llmkube/python-tree.sha256
if [ "${DSV41_VERIFY_TREE:-0}" = "1" ]; then
    [ -f "$STAMP" ] || die "no build stamp at $STAMP; this is not a properly built image"
    want="$(cat "$STAMP")"
    got="$(find "$SITE/vllm" "$SITE/b12x" -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
    [ "$want" = "$got" ] || die "vllm+b12x tree digest mismatch: built=${want} now=${got}; something modified site-packages after build"
    say "vllm+b12x tree digest verified: ${got}"
fi
[ "$#" -gt 0 ] || die "no command given; the InferenceService supplies the vllm serve invocation"
say "exec: $*"
exec "$@"
