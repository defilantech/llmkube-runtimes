#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Gufo ROCm (gfx1151) image gate (cheap, no GPU required).
#
# rocm-gufo/Dockerfile already guards the build tree and the final stage. This
# gate re-asserts the same guarantees against the SHIPPED image from outside,
# because the failures we care about survive a green build log: a stale COPY,
# a label that names one commit while the binary is another, or an FFmpeg
# package pulled in by some later dependency change.
#
#   1. pin      the image records the Dockerfile's GUFO_SHA in three places
#               that agree: /opt/llmkube/gufo-commit, the io.llmkube.gufo.sha
#               label, and `gufo --version`.
#   2. runs     `gufo --help` and `gufo --version` exit 0; every linked library
#               resolves.
#   3. flags    `gufo serve help llm` lists every flag the bake-off serve
#               command passes.
#   4. ffmpeg   no FFmpeg library linked, no ffmpeg/ffprobe executable, no
#               FFmpeg package, no libav*/libsw* file anywhere in the image.
#   5. license  Gufo's MIT text, NOTICE and THIRD_PARTY_NOTICES.md present in
#               both /opt/llmkube and Gufo's own install location, and equal.
#
# Real GPU offload, a model load, and MTP acceptance are Tier-2 on gfx1151.
set -euo pipefail

IMAGE="${1:?usage: rocm-gufo-gate.sh <image-ref>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${HERE}/../rocm-gufo/Dockerfile"

run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

echo "== 1. pinned commit recorded and consistent =="
want_sha="$(sed -n 's/^ARG GUFO_SHA=\([0-9a-f]\{40\}\)$/\1/p' "${DOCKERFILE}")"
want_ref="$(sed -n 's/^ARG GUFO_REF=\(v[0-9][0-9.]*\)$/\1/p' "${DOCKERFILE}")"
if [ -z "${want_sha}" ] || [ -z "${want_ref}" ]; then
  echo "FAIL: could not read a full ARG GUFO_SHA / ARG GUFO_REF from ${DOCKERFILE}"
  exit 1
fi
file_sha="$(run cat /opt/llmkube/gufo-commit)"
label_sha="$(docker inspect --format '{{ index .Config.Labels "io.llmkube.gufo.sha" }}' "${IMAGE}")"
version_line="$(run /usr/local/bin/gufo --version)"
want_version="gufo version ${want_ref#v} (${want_sha})"
echo "Dockerfile:            ${want_ref} ${want_sha}"
echo "/opt/llmkube:          ${file_sha}"
echo "label:                 ${label_sha}"
echo "gufo --version:        ${version_line}"
rc=0
[ "${file_sha}" = "${want_sha}" ] || { echo "FAIL: /opt/llmkube/gufo-commit does not match the Dockerfile pin"; rc=1; }
[ "${label_sha}" = "${want_sha}" ] || { echo "FAIL: io.llmkube.gufo.sha label does not match the Dockerfile pin"; rc=1; }
[ "${version_line}" = "${want_version}" ] || { echo "FAIL: gufo --version is not '${want_version}'"; rc=1; }
[ "${rc}" = 0 ] || exit 1
echo "PASS: pin ${want_sha} recorded in file, label and binary"

echo "== 2. binary runs and every library resolves =="
# Every grep below runs INSIDE the container against a FILE, never against a
# pipe from this shell: `cmd | grep -q` lets grep exit at the first match, the
# writer takes SIGPIPE, and under pipefail a passing check reports failure
# (the hazard documented in scripts/qwen4exp-gate.sh).
if ! run sh -c '
      /usr/local/bin/gufo --help > /tmp/help.txt 2>&1 || { echo "FAIL: gufo --help exited non-zero"; cat /tmp/help.txt; exit 1; }
      grep -q "serve" /tmp/help.txt || { echo "FAIL: gufo --help does not list the serve command"; exit 1; }
      ldd /usr/local/bin/gufo > /tmp/ldd.txt 2>&1
      test -s /tmp/ldd.txt || { echo "FAIL: ldd produced no output"; exit 1; }
      if grep -q "not found" /tmp/ldd.txt; then echo "FAIL: unresolved libraries:"; grep "not found" /tmp/ldd.txt; exit 1; fi
      for lib in libamdhip64 libhipblas libhipblaslt librocblas libicuuc libcurl libcrypto libpng libjpeg libwebp; do
        grep -q "${lib}" /tmp/ldd.txt || { echo "FAIL: gufo does not link ${lib}; the build is not the HIP release build"; exit 1; }
      done
    '; then
  exit 1
fi
echo "PASS: gufo --help runs; HIP, hipBLAS, hipBLASLt, rocBLAS and system libraries all resolve"

echo "== 3. serve llm accepts the bake-off flags =="
# `gufo serve help llm` prints the llm parser's help and loads nothing
# (src/cli/serve/serve.cpp, PrintServeHelp). The bake-off command passes each
# of these; a parser without one aborts at startup on the node.
if ! run sh -c '
      /usr/local/bin/gufo serve help llm > /tmp/llm-help.txt 2>&1 || { echo "FAIL: gufo serve help llm exited non-zero"; cat /tmp/llm-help.txt; exit 1; }
      rc=0
      for flag in --host --port --sessions --model --context --speculative --mtp-model --draft-tokens; do
        grep -q -- "${flag}" /tmp/llm-help.txt || { echo "FAIL: gufo serve llm does not list ${flag}"; rc=1; }
      done
      grep -q "mtp" /tmp/llm-help.txt || { echo "FAIL: --speculative help does not offer mtp"; rc=1; }
      exit $rc
    '; then
  exit 1
fi
echo "PASS: --host --port --sessions --model --context --speculative mtp --mtp-model --draft-tokens accepted"

echo "== 4. no FFmpeg anywhere =="
if ! run sh -c '
      rc=0
      ldd /usr/local/bin/gufo > /tmp/ldd.txt 2>&1
      if grep -Eiq "libav(codec|format|util|filter|device)|libsw(scale|resample)|libpostproc" /tmp/ldd.txt; then
        echo "FAIL: gufo links an FFmpeg library"; rc=1
      fi
      if command -v ffmpeg >/dev/null 2>&1 || command -v ffprobe >/dev/null 2>&1; then
        echo "FAIL: ffmpeg/ffprobe is on PATH"; rc=1
      fi
      find / -xdev \( -name ffmpeg -o -name ffprobe \) -type f > /tmp/ff-bin.txt 2>/dev/null || true
      find / -xdev \( -name "libavcodec*" -o -name "libavformat*" -o -name "libavutil*" \
        -o -name "libavfilter*" -o -name "libavdevice*" -o -name "libswscale*" \
        -o -name "libswresample*" -o -name "libpostproc*" \) > /tmp/ff-lib.txt 2>/dev/null || true
      if [ -s /tmp/ff-bin.txt ]; then echo "FAIL: FFmpeg executables in the image:"; cat /tmp/ff-bin.txt; rc=1; fi
      if [ -s /tmp/ff-lib.txt ]; then echo "FAIL: FFmpeg libraries in the image:"; cat /tmp/ff-lib.txt; rc=1; fi
      dpkg-query -W -f="\${Package}\n" > /tmp/pkgs.txt
      if grep -Eq "^(ffmpeg|libav(codec|format|util|filter|device)|libsw(scale|resample)|libpostproc)" /tmp/pkgs.txt; then
        echo "FAIL: FFmpeg packages installed:"
        grep -E "^(ffmpeg|libav(codec|format|util|filter|device)|libsw(scale|resample)|libpostproc)" /tmp/pkgs.txt
        rc=1
      fi
      exit $rc
    '; then
  exit 1
fi
echo "PASS: no FFmpeg library linked, no executable, no library file, no package"

echo "== 5. license and notice files =="
if ! run sh -c '
      rc=0
      for f in /opt/llmkube/NOTICE /opt/llmkube/LICENSE.gufo-MIT /opt/llmkube/THIRD_PARTY_NOTICES.md \
               /opt/llmkube/linked-libs.txt /opt/llmkube/runtime-packages.txt \
               /usr/local/share/licenses/gufo/LICENSE /usr/local/share/licenses/gufo/NOTICE \
               /usr/local/share/licenses/gufo/THIRD_PARTY_NOTICES.md \
               /usr/local/share/licenses/gufo/third-party/LICENSE.ds4 \
               /usr/local/share/licenses/gufo/third-party/llama.cpp.txt \
               /usr/local/share/licenses/gufo/third-party/hipcub.txt; do
        test -s "$f" || { echo "FAIL: $f missing or empty"; rc=1; }
      done
      head -1 /opt/llmkube/LICENSE.gufo-MIT > /tmp/l1.txt
      grep -qx "MIT License" /tmp/l1.txt || { echo "FAIL: LICENSE.gufo-MIT is not the MIT text"; rc=1; }
      cmp -s /opt/llmkube/LICENSE.gufo-MIT /usr/local/share/licenses/gufo/LICENSE \
        || { echo "FAIL: vendored LICENSE.gufo-MIT differs from the one Gufo installed"; rc=1; }
      cmp -s /opt/llmkube/THIRD_PARTY_NOTICES.md /usr/local/share/licenses/gufo/THIRD_PARTY_NOTICES.md \
        || { echo "FAIL: vendored THIRD_PARTY_NOTICES.md differs from the one Gufo installed"; rc=1; }
      # Case-sensitive on the AGPL title line on purpose: Gufo retains the
      # GPL-3.0 text (third-party/ffmpeg-GPL-3.0.txt), whose section 13 reads
      # "Use with the GNU Affero General Public License", so a
      # case-insensitive "GNU AFFERO" grep fails on a clean image.
      # grep exits 0 on a match, 1 on none, 2 on an error (a missing directory, unreadable file); only 1 is clean.
      agpl_rc=0
      grep -rl "GNU AFFERO GENERAL PUBLIC LICENSE" /opt/llmkube /usr/local/share/licenses/gufo > /tmp/agpl.txt || agpl_rc=$?
      if [ "$agpl_rc" -eq 0 ]; then
        echo "FAIL: AGPL text found:"; cat /tmp/agpl.txt; rc=1
      elif [ "$agpl_rc" -ne 1 ]; then
        echo "FAIL: the AGPL scan could not read its directories (grep exit $agpl_rc)"; rc=1
      fi
      # A library with no owning package has no /usr/share/doc copyright file
      # we can point to. Loud, but a receipt-format surprise should not fail a
      # multi-hour HIP build; the finding goes to the license follow-up.
      if grep -q UNOWNED /opt/llmkube/linked-libs.txt; then
        echo "WARN: linked libraries with no owning package (license follow-up):"
        grep UNOWNED /opt/llmkube/linked-libs.txt
      fi
      exit $rc
    '; then
  exit 1
fi
echo "PASS: MIT text, NOTICE, THIRD_PARTY_NOTICES.md and third-party texts present and consistent; no AGPL text"

echo "== 6. no development tools shipped =="
if ! run sh -c 'test ! -e /usr/local/bin/gufo-kernel-bench && test ! -e /usr/local/bin/tune_hipblaslt'; then
  echo "FAIL: Gufo development tools are installed (GUFO_BUILD_TOOLS must be OFF)"
  exit 1
fi
echo "PASS: production install only"

echo "== verdict =="
echo "PASS: ${IMAGE} is Gufo ${want_ref} (${want_sha}), packaged correctly, no FFmpeg"
