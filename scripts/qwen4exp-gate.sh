#!/usr/bin/env bash
#
# Qwen3.8-Flash-Next (qwen4exp) variant gate (cheap, no GPU required).
#
# vulkan-qwen4exp/Dockerfile already guards the BUILD TREE: the arch compiled
# in, the qwen4exp MTP draft graph compiled in, the argument parser accepting
# the flags the InferenceService passes, and the Vulkan backend resolving every
# relocation at dlopen. This gate re-checks the first three against the FINAL
# IMAGE, because the failure mode we care about survives the COPY: a typo in
# `COPY --from=build /out/ /app/` ships the stock runtime, and the resulting
# pod loads Qwen3.8-Flash-Next, generates correctly, and does it at the target's
# own speed with no error anywhere. That reads as a hardware result, not a
# packaging bug, and it is exactly the finding this experiment would then
# report wrongly.
#
# Runs the image's own binary and files only. Real GPU offload, a load, and the
# `draft acceptance =` line are Tier-2 on gfx1151, and are acceptance steps in
# the research kit rather than CI.
set -euo pipefail

IMAGE="${1:?usage: qwen4exp-gate.sh <image-ref>}"

run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

echo "== qwen4exp arch compiled in =="
if ! run sh -c 'grep -rqa qwen4exp /app/'; then
  echo "FAIL: qwen4exp arch string absent from the shipped image"
  exit 1
fi
if ! run sh -c 'grep -rqa ple_conv1d /app/'; then
  echo "FAIL: PLE tensor names absent; the arch registered without its weights"
  exit 1
fi
echo "PASS: qwen4exp arch and PLE tensors present"

echo "== qwen4exp MTP draft graph compiled in =="
# "QWEN4EXP MTP:" is a throw message ggml-org#28243 adds to
# src/models/qwen4exp.cpp. Without it the image serves this model with
# speculation silently off.
if ! run sh -c "grep -rqa 'QWEN4EXP MTP:' /app/"; then
  echo "FAIL: the qwen4exp MTP draft graph is absent from the shipped image."
  echo "      Speculation would be silently off and the throughput result wrong."
  exit 1
fi
echo "PASS: qwen4exp MTP draft graph present"

echo "== argument parser accepts the flags the InferenceService renders =="
# spec.speculativeDecoding renders --spec-type draft-mtp and
# --spec-draft-n-max; a parser without them aborts llama-server at startup.
#
# The greps run INSIDE the container and against a FILE, never against a pipe
# from this shell. `printf ... | grep -q` looks equivalent and is not: grep
# exits at the first match, the writer takes SIGPIPE, and under `set -o
# pipefail` the pipeline then reports failure on a PASSING build. It also fails
# only sometimes, which is worse than always: this check passed on an arm64
# developer daemon and failed on the amd64 runner, because the help text fit one
# pipe buffer in the first case and not the second. Same hazard the Dockerfile's
# arg-surface guard documents.
if ! run sh -c '
      /app/llama-server --help > /tmp/qwen4exp-help.txt 2>&1 || true
      test -s /tmp/qwen4exp-help.txt || { echo "FAIL: llama-server --help produced no output"; exit 1; }
      rc=0
      for flag in --spec-type --spec-draft-n-max; do
        grep -q -- "$flag" /tmp/qwen4exp-help.txt || { echo "FAIL: llama-server --help does not list $flag"; rc=1; }
      done
      exit $rc
    '; then
  exit 1
fi
echo "PASS: --spec-type and --spec-draft-n-max accepted"

echo "== verdict =="
echo "PASS: Qwen3.8-Flash-Next variant image is packaged correctly"
