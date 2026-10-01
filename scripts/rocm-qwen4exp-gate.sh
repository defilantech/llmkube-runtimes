#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# rocm-qwen4exp variant gate (cheap, no GPU required).
#
# rocm-qwen4exp/Dockerfile guards the BUILD TREE. This gate checks the FINAL
# IMAGE, because the failures we care about survive or appear at the COPY into
# the slim runtime stage:
#
#   1. The binaries run: llama-server --version and llama-perplexity --version.
#   2. The HIP backend resolves every relocation at dlopen (RTLD_NOW, the #725
#      guard adapted to libggml-hip.so), using the runtime stage's ROCm
#      libraries rather than the builder's full SDK. A runtime library missing
#      from the slim stage (libhipblas.so.3 was one, see rocm/Dockerfile) only
#      shows up here; on a node it reads as "no usable GPU found" and a silent
#      CPU fallback.
#   3. llama-server enumerates its backends without a loader error.
#   4. patches/APPLIED.md matches the patches: every rocm-qwen4exp/patches/*.patch
#      has exactly one row with the right sha256, the image carries the same
#      files, and every row's marker string is in the image's libllama.so.
#
# The qwen4exp arch, MTP draft graph and flag-surface checks are
# scripts/qwen4exp-gate.sh, which the workflow runs against this image too.
# Real GPU offload is Tier-2 on gfx1151.
set -euo pipefail

IMAGE="${1:?usage: rocm-qwen4exp-gate.sh <image-ref>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/rocm-qwen4exp/patches"
MANIFEST="${PATCH_DIR}/APPLIED.md"
IMG_PATCH_DIR="/usr/share/llmkube/rocm-qwen4exp/patches"
DLCHECK="/usr/libexec/llmkube/dlcheck"

fail() { echo "FAIL: $*"; exit 1; }

run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

echo "== binaries run =="
for bin in llama-server llama-perplexity; do
  if ! out="$(run "/app/${bin}" --version 2>&1)"; then
    echo "${out}"
    fail "${bin} --version exited non-zero"
  fi
  echo "${out}" | tail -n 2
done
echo "PASS: llama-server and llama-perplexity run"

echo "== HIP backend dlopen(RTLD_NOW) in the runtime image =="
# LD_LIBRARY_PATH comes from the image ENV (/opt/rocm/lib:/app). The checker
# is built in the image's build stage (same glibc as the runtime base), so the
# gate needs no host compiler and runs from any docker host.
if ! out="$(run "${DLCHECK}" /app/libggml-hip.so 2>&1)"; then
  echo "${out}"
  fail "libggml-hip.so does not load in the runtime image; a ROCm runtime library is missing or a symbol is unresolved"
fi
echo "${out}"
echo "${out}" | grep -q '^dlopen OK: ' || fail "dlopen checker gave no verdict"
echo "PASS: HIP backend resolves every relocation at dlopen"

echo "== llama-server --list-devices =="
# A GPU-less runner legitimately enumerates no ROCm device and logs
# "failed to initialize ROCm: no ROCm-capable device", so this gates on loader
# errors only, never on a positive device.
if ! out="$(run /app/llama-server --list-devices 2>&1)"; then
  echo "${out}"
  fail "llama-server --list-devices exited non-zero"
fi
echo "${out}"
if echo "${out}" | grep -qiE 'symbol lookup error|undefined symbol|cannot open shared object|failed to load|failed to find ggml_backend_init|returned NULL|incompatible API version'; then
  fail "backend load error in the runtime image"
fi
echo "PASS: backends enumerate without a loader error"

echo "== APPLIED.md matches the patches =="
[ -f "${MANIFEST}" ] || fail "${MANIFEST} is missing"

# Rows look like: | 0010-name.patch | upstream | head | author | scope | form | marker | sha256 |
rows="$(grep -E '^\| [0-9]{4}-[^|]*\.patch \|' "${MANIFEST}" || true)"
[ -n "${rows}" ] || fail "no patch rows found in APPLIED.md"

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }

manifest_files=()
rc=0
while IFS= read -r row; do
  IFS='|' read -r _ c_file _ _ _ _ _ c_marker c_sha _ <<<"${row}"
  file="$(trim "${c_file}")"
  marker="$(trim "${c_marker}")"
  want="$(trim "${c_sha}")"
  manifest_files+=("${file}")
  row_ok=1

  if [ ! -f "${PATCH_DIR}/${file}" ]; then
    echo "FAIL: APPLIED.md lists ${file}, which is not in rocm-qwen4exp/patches/"; rc=1; continue
  fi
  got="$(sha256_of "${PATCH_DIR}/${file}")"
  if [ "${got}" != "${want}" ]; then
    echo "FAIL: ${file} sha256 ${got} does not match APPLIED.md (${want})"; rc=1; row_ok=0
  fi
  img="$(run sh -c 'sha256sum "$1" 2>/dev/null | cut -d" " -f1' sh "${IMG_PATCH_DIR}/${file}")"
  if [ "${img}" != "${want}" ]; then
    echo "FAIL: the image's copy of ${file} (${img:-missing}) does not match APPLIED.md (${want})"; rc=1; row_ok=0
  fi
  if [ -z "${marker}" ]; then
    echo "FAIL: ${file} has no marker in APPLIED.md"; rc=1
  elif ! run sh -c 'grep -qaF -- "$1" /app/libllama.so' sh "${marker}"; then
    echo "FAIL: marker '${marker}' for ${file} is absent from the image's libllama.so"; rc=1
  elif [ "${row_ok}" = 1 ]; then
    echo "ok: ${file} (sha256 matches in repo and image, marker '${marker}' present)"
  fi
done <<<"${rows}"

# Every patch on disk, and every patch in the image, must have a row.
for f in "${PATCH_DIR}"/*.patch; do
  base="$(basename "$f")"
  found=0
  for m in "${manifest_files[@]}"; do [ "$m" = "$base" ] && found=1; done
  [ "$found" = 1 ] || { echo "FAIL: ${base} is in rocm-qwen4exp/patches/ but has no row in APPLIED.md"; rc=1; }
done
img_list="$(run sh -c 'cd "$1" && ls -1 -- *.patch' sh "${IMG_PATCH_DIR}")"
while IFS= read -r base; do
  [ -n "$base" ] || continue
  found=0
  for m in "${manifest_files[@]}"; do [ "$m" = "$base" ] && found=1; done
  [ "$found" = 1 ] || { echo "FAIL: the image carries ${base}, which has no row in APPLIED.md"; rc=1; }
done <<<"${img_list}"

[ "${rc}" = 0 ] || exit 1
echo "PASS: APPLIED.md, the repo patches, the image's patches and libllama.so agree"

echo "== verdict =="
echo "PASS: rocm-qwen4exp image is packaged correctly"
