#!/bin/bash
#
# LLMKube GB10 native ExLlamaV3 launcher.
#
# Two checks, both fail-closed, then exec the server. Neither is decoration:
#
#  1. Tree digest. EXL3_VERIFY_TREE=1 recomputes the digest of the installed
#     exllamav3 package and compares it to the stamp written at build. A
#     site-packages mutated after the build (a bind-mounted patch, a pip install
#     in a running container) is then caught here rather than served.
#
#  2. MemAvailable pre-flight. The ATS loader aliases ~107 GiB of weights out of
#     the page cache. If the box cannot supply that, the failure surfaces deep
#     inside CUDA with a less useful error; failing at the door with the number
#     is the whole point of the check. The floor is deliberately a knob: it is a
#     policy about the box, not a property of the image.
#
# The SCAFFOLD server is started here. It binds the port and answers liveness; it
# does not load a model. The OpenAI-compatible implementation replaces
# server/app.py, and this launcher does not change for it.
set -euo pipefail

say() { echo "[exl3] $*"; }
die() { echo "[exl3] FATAL: $*" >&2; exit 1; }

EXL3_TREE="${EXL3_TREE:-$(ls -d /opt/venv/lib/python*/site-packages/exllamav3 2>/dev/null | head -1)}"
[ -n "${EXL3_TREE}" ] || die "exllamav3 package not found under /opt/venv/lib/python*/site-packages"

# Keep this command identical to build/receipt.sh; the stamp is only meaningful
# if both sides hash the same set of files the same way.
tree_digest() {
    find "${EXL3_TREE}" -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}

if [ "${EXL3_VERIFY_TREE:-0}" = "1" ]; then
    stamp_file=/opt/llmkube/exllamav3-tree.sha256
    [ -f "${stamp_file}" ] || die "no build stamp at ${stamp_file}; this is not a properly built image"
    want="$(cat "${stamp_file}")"
    got="$(tree_digest)"
    [ "${want}" = "${got}" ] || die "exllamav3 tree digest mismatch: built=${want} now=${got}; site-packages changed after build"
    say "exllamav3 tree digest verified: ${got}"
fi

min_gib="${EXL3_MIN_MEM_AVAILABLE_GIB:-115}"
avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[ -n "${avail_kb}" ] || die "cannot read MemAvailable from /proc/meminfo"
avail_gib=$(( avail_kb / 1024 / 1024 ))
if [ "${avail_gib}" -lt "${min_gib}" ]; then
    die "MemAvailable ${avail_gib} GiB is below the ${min_gib} GiB floor (EXL3_MIN_MEM_AVAILABLE_GIB)."$'\n'"The ATS loader aliases ~107 GiB of weights out of page cache, so this box cannot load the model as-is."
fi
say "MemAvailable ${avail_gib} GiB >= ${min_gib} GiB floor"

port="${EXL3_PORT:-5000}"
# Args passthrough: the Tier-1 gate runs `entrypoint.sh true` to exercise the
# checks above without starting a server. A normal start passes no args.
if [ "$#" -gt 0 ]; then
    say "exec: $*"
    exec "$@"
fi
say "starting scaffold server on :${port} (model dir: ${EXL3_MODEL_DIR:-unset})"
exec python3 -m uvicorn server.app:app --host 0.0.0.0 --port "${port}"
