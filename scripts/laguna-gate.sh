#!/usr/bin/env bash
#
# Laguna variant gate (cheap, no GPU required).
#
# vulkan-laguna/Dockerfile already guards the BUILD TREE: laguna arch present,
# fork autoparser compiled in, shipped template carries the argument encoding.
# This gate re-checks the same properties against the FINAL IMAGE, because the
# failure mode we care about is silent: a mistaken COPY path or a stale ENV
# default produces an image that builds green, serves Laguna at full speed, and
# emits zero tool calls. That looks like a model quality problem, not a
# packaging bug, so it is worth catching in CI rather than in an agent loop.
#
# Runs the image's own binary and files only. Real GPU offload and a live
# tool-call round trip are Tier-2 (the out-of-band gfx1151 promoter).
set -euo pipefail

IMAGE="${1:?usage: laguna-gate.sh <image-ref>}"

run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

echo "== laguna arch compiled in =="
if ! run sh -c 'grep -rqa laguna /app/'; then
  echo "FAIL: laguna arch string absent from the shipped image"
  exit 1
fi
echo "PASS: laguna arch present"

echo "== fork tool-call autoparser compiled in =="
# Pure upstream llama.cpp cannot parse poolside tool calls. If this string is
# missing we built upstream, not TheTom's fork, and the image is pointless.
if ! run sh -c 'grep -rqa "differential autoparser" /app/'; then
  echo "FAIL: differential autoparser absent; this is not TheTom's laguna/port build"
  exit 1
fi
echo "PASS: fork autoparser present"

echo "== default chat template resolves and carries the argument encoding =="
# The autoparser derives the tool-call format from the template. A template
# without arg_key yields a parser that drops every tool call, which is exactly
# what poolside's own released GGUF embeds. Verify the image's ENV default
# points at a real file AND that the file is the corrected one.
tmpl="$(docker run --rm --entrypoint sh "${IMAGE}" -c 'echo "${LLAMA_ARG_CHAT_TEMPLATE_FILE:-}"')"
if [ -z "${tmpl}" ]; then
  echo "FAIL: LLAMA_ARG_CHAT_TEMPLATE_FILE is not set in the image"
  exit 1
fi
echo "default template: ${tmpl}"
if ! run sh -c "test -f '${tmpl}'"; then
  echo "FAIL: LLAMA_ARG_CHAT_TEMPLATE_FILE points at ${tmpl}, which does not exist in the image"
  exit 1
fi
if ! run sh -c "grep -qa arg_key '${tmpl}'"; then
  echo "FAIL: ${tmpl} lacks arg_key; tool calls would silently not parse"
  exit 1
fi
echo "PASS: default template exists and carries arg_key"

echo "== verdict =="
echo "PASS: Laguna variant image is packaged correctly"
