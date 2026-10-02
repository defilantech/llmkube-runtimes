# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
#
# LLMKube AMD/ROCm (HIP) Gufo runtime for gfx1151 (Strix Halo).
#
# WHY THIS IMAGE EXISTS
#
# Gufo (gufo-org/gufo, MIT, C++/HIP) is a native gfx1151 inference engine that
# reads exactly the Qwen3.8-Flash-Next files the lab already stages: unsloth
# UD-Q4_K_XL plus the `shared-` Q8_0 MTP head. It is a bake-off candidate
# against the Vulkan llama.cpp image, not (yet) a production tier.
#
# Gufo publishes its own container (ghcr.io/gufo-org/toolboxes/gufo-runtime,
# Nix-built). We do not serve it: runtime images are built here from a pinned
# source on a pinned base, so this recipe rebuilds Gufo from the release tag's
# commit against AMD's ROCm 7.2.3 Ubuntu image, the toolchain Gufo's README
# qualifies ("GCC 15.3 and ROCm 7.2.3").
#
# HOW IT WILL BE SERVED (the bake-off, not asserted by CI):
#
#   gufo serve --host 0.0.0.0 --port 8080 llm \
#     --model <UD-Q4_K_XL shard 1> \
#     --speculative mtp --mtp-model <mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf> \
#     --sessions 2 --context 262144
#
# Note the MTP head is the `shared-` one, the opposite of vulkan-qwen4exp:
# Gufo's --mtp-model help text names "the mtp-...-shared-*.gguf sidecar".
#
# WHAT CI CAN AND CANNOT PROVE
#
# The hosted runner has no GPU. The guards below prove the binary was built at
# the pinned commit, links against everything it needs, starts, parses the
# flags the serve command above uses, and contains no FFmpeg. They prove
# nothing about gfx1151 kernels, MTP acceptance, or speed; those are Tier-2
# acceptance steps on shadowstrix.
#
# FFMPEG IS NOT IN THIS IMAGE
#
# Gufo never links FFmpeg (no libav* include or link anywhere in the source at
# the pin); it invokes `ffmpeg`/`ffprobe` as separate executables, and only for
# the video/TTS media paths (THIRD_PARTY_NOTICES.md, "FFmpeg is invoked as a
# separate process, not linked into Gufo"). Text serving never calls it. So the
# build stage omits the `ffmpeg` apt package Gufo's README lists, the runtime
# stage never installs it, and both the build and the gate assert its absence
# (no libav*/libsw* in ldd, no ffmpeg/ffprobe executable, no FFmpeg package).
# Media endpoints in this image will fail by design.
#
# PROVENANCE AND LICENSE
#
# Gufo is MIT; its adapted code (llama.cpp/ggml, ds4, h3.c) is MIT and
# BSD-3-Clause, and the linked ROCm libraries are MIT/BSD-3-Clause. See NOTICE.
# LICENSE.gufo-MIT and THIRD_PARTY_NOTICES.md are byte-for-byte copies of the
# files at the pinned commit; the build fails if they drift.

# Gufo, pinned by release tag AND the exact commit it must resolve to. v0.5.0 is
# a lightweight tag on 23cacbb9 ("chore(main): release 0.5.0", 2026-10-02),
# verified with `gh api repos/gufo-org/gufo/git/ref/tags/v0.5.0`. It is the
# first release with the thinking-on prompt-cache fix (gufo-org/gufo#335, fixed
# by #362, #358, #369 and #348). That fix was validated on Strix Halo at
# 93af45d4, which differs from the release commit only by a functional-test
# change (#378) and the release bump; no server code changed between them.
ARG GUFO_REPO=https://github.com/gufo-org/gufo.git
ARG GUFO_REF=v0.5.0
ARG GUFO_SHA=23cacbb9379d5a8e8531535270891729ca6c00dd
# AMD's ROCm dev image, the plain (not -complete) 7.2.3 tag, pinned by digest.
# Single-arch amd64 manifest (application/vnd.docker.distribution.manifest.v2+json)
# read from registry-1.docker.io on 2026-09-30. It carries rocm-dev (HIP clang,
# CLR, device libs), build-essential and AMD's 7.2.3 apt repo; the libraries
# Gufo links are added below from that same repo.
ARG ROCM_VERSION=7.2.3
ARG ROCM_DEV_DIGEST=sha256:8ab378c7264c7e15496f548392c518a6b2feeb322d6233364c6c2ebc8e7ffb42
# ubuntu:24.04 (the dev image's own base release), OCI index digest read from
# registry-1.docker.io on 2026-09-30.
ARG UBUNTU_DIGEST=sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
# Gufo's CMakeLists hard-fails on anything else; stated here for readability.
ARG GPU_TARGETS=gfx1151
# HIP compiles are memory-heavy. Gufo's README uses --parallel 4; the workflow
# derives this from the runner.
ARG MAX_JOBS=4

FROM docker.io/rocm/dev-ubuntu-24.04:${ROCM_VERSION}@${ROCM_DEV_DIGEST} AS build
ARG GUFO_REPO
ARG GUFO_REF
ARG GUFO_SHA
ARG GPU_TARGETS
ARG MAX_JOBS
# Gufo README "Without Nix": build-essential cmake ninja-build pkg-config
# libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev
# ffmpeg, then hipblas-dev hipblaslt-dev rocblas-dev hipcub-dev rocprim-dev
# rocwmma-dev. Everything except `ffmpeg` (see header). build-essential is
# already in the base; git is for the pinned clone. libwebp-dev is new since
# v0.3.0: CMakeLists.txt requires it via pkg_check_modules(WEBP REQUIRED).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git ca-certificates cmake ninja-build pkg-config \
      libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev \
      hipblas-dev hipblaslt-dev rocblas-dev hipcub-dev rocprim-dev rocwmma-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${GUFO_REF}" "${GUFO_REPO}" . \
    && got="$(git rev-parse HEAD)" \
    && if [ "${got}" != "${GUFO_SHA}" ]; then \
         echo "ERROR: ${GUFO_REF} resolved to ${got}, expected ${GUFO_SHA}"; exit 1; \
       fi \
    && ver="$(cat version.txt)" \
    && if [ "v${ver}" != "${GUFO_REF}" ]; then \
         echo "ERROR: version.txt says ${ver}, which does not match ${GUFO_REF}"; exit 1; \
       fi \
    && echo "PASS: ${GUFO_REF} is ${GUFO_SHA} (version.txt ${ver})"
# The attribution files we vendor must be the ones at the pin. A byte compare
# here turns a pin bump that forgot to refresh them into a build failure.
COPY LICENSE.gufo-MIT THIRD_PARTY_NOTICES.md /vendored/
RUN cmp /vendored/LICENSE.gufo-MIT /src/LICENSE \
    && cmp /vendored/THIRD_PARTY_NOTICES.md /src/THIRD_PARTY_NOTICES.md \
    && echo "PASS: vendored LICENSE and THIRD_PARTY_NOTICES.md match ${GUFO_SHA}"
# Gufo's own production configuration, unmodified: the `release` preset
# (CMakePresets.json: Ninja, RelWithDebInfo, BUILD_TESTING=OFF,
# ENGINE_ENABLE_HIP=ON), with only what .devops/nix/package.nix also passes:
#   GUFO_RELEASE_VERSION / GUFO_REVISION  -> `gufo --version` names the pin.
#     Nix passes the 7-char shortRev; we pass the full SHA so the version line
#     alone identifies the commit.
#   CMAKE_HIP_COMPILER                    -> ROCm's clang, as Nix passes its own.
# The ffmpeg/ffprobe executable names stay at their defaults ("ffmpeg" on
# PATH), which resolves to nothing in this image. CMAKE_INSTALL_PREFIX is
# /usr/local because Gufo bakes install paths (eval data, H3 manifest) into the
# binary, so the prefix must be where the files actually land.
RUN ver="${GUFO_REF#v}" \
    && cmake --preset release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_HIP_ARCHITECTURES="${GPU_TARGETS}" \
      -DGUFO_RELEASE_VERSION="${ver}" \
      -DGUFO_REVISION="${GUFO_SHA}" \
    && cmake --build --preset release --parallel "${MAX_JOBS}" \
    && DESTDIR=/out cmake --install build/release
# Gufo's own install check (.devops/nix/package.nix installCheckPhase), run on
# the installed tree: exact version line from both names, help runs, no
# development tools installed, license files installed. None of it touches the
# GPU; Nix runs the same checks in a GPU-less sandbox.
RUN export LD_LIBRARY_PATH=/opt/rocm/lib \
    && bin=/out/usr/local/bin \
    && want="gufo version ${GUFO_REF#v} (${GUFO_SHA})" \
    && test "$("${bin}/gufo" --version)" = "${want}" \
    && test "$("${bin}/gufo-server" --version)" = "${want}" \
    && "${bin}/gufo" --help > /dev/null \
    && "${bin}/gufo-server" --help > /dev/null \
    && test ! -e "${bin}/gufo-kernel-bench" \
    && test -f /out/usr/local/share/licenses/gufo/LICENSE \
    && test -f /out/usr/local/share/licenses/gufo/third-party/LICENSE.ds4 \
    && echo "PASS: Gufo install check (${want})"
# FFMPEG GUARD, build side. The binary must not link any FFmpeg library. ldd is
# written to a file and grepped, never piped into grep -q (SIGPIPE under
# pipefail inverts a passing check; see vulkan-qwen4exp/Dockerfile).
RUN LD_LIBRARY_PATH=/opt/rocm/lib ldd /out/usr/local/bin/gufo > /tmp/ldd.txt 2>&1; \
    if [ ! -s /tmp/ldd.txt ]; then echo "ERROR: ldd produced no output"; exit 1; fi; \
    if grep -Eiq 'libav(codec|format|util|filter|device)|libsw(scale|resample)|libpostproc' /tmp/ldd.txt; then \
      echo "ERROR: gufo links an FFmpeg library:"; cat /tmp/ldd.txt; exit 1; \
    fi; \
    if grep -q 'not found' /tmp/ldd.txt; then \
      echo "ERROR: unresolved libraries in the build stage:"; cat /tmp/ldd.txt; exit 1; \
    fi; \
    echo "PASS: gufo links no FFmpeg library"; cat /tmp/ldd.txt

FROM docker.io/ubuntu:24.04@${UBUNTU_DIGEST} AS server
ARG GUFO_REPO
ARG GUFO_REF
ARG GUFO_SHA
ARG ROCM_VERSION
LABEL org.opencontainers.image.title="llmkube-gufo-rocm-gfx1151" \
      org.opencontainers.image.description="LLMKube AMD/ROCm Gufo runtime for gfx1151 (Strix Halo), built from the pinned Gufo release commit. Text serving only; no FFmpeg." \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      io.llmkube.gufo.repo="${GUFO_REPO}" \
      io.llmkube.gufo.ref="${GUFO_REF}" \
      io.llmkube.gufo.sha="${GUFO_SHA}" \
      io.llmkube.rocm.version="${ROCM_VERSION}"
# Runtime libraries only, matching Gufo's link line (CMakeLists.txt:
# ICU uc/i18n, libcurl, OpenSSL Crypto, libpng, libjpeg, libwebp, OpenMP,
# hip::host, roc::hipblas, roc::hipblaslt, roc::rocblas). ROCm comes from AMD's apt repo at
# the same ROCM_VERSION as the builder. curl stays for the HEALTHCHECK; gnupg is
# only for the repo key and is purged. No ffmpeg.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
       > /etc/apt/sources.list.d/rocm.list \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      hip-runtime-amd hipblas hipblaslt rocblas \
      libicu74 libcurl4t64 libssl3t64 libpng16-16t64 libjpeg-turbo8 libwebp7 libgomp1 \
    && apt-get purge -y gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
# The installed tree: /usr/local/bin/gufo (+ the gufo-server symlink),
# /usr/local/share/gufo (data paths baked into the binary) and
# /usr/local/share/licenses/gufo (LICENSE, NOTICE, THIRD_PARTY_NOTICES.md and
# every third-party license text, installed by Gufo's own CMake rules).
COPY --from=build /out/usr/local/ /usr/local/
COPY NOTICE LICENSE.gufo-MIT THIRD_PARTY_NOTICES.md /opt/llmkube/
ENV LD_LIBRARY_PATH=/opt/rocm/lib \
    GUFO_HOST=0.0.0.0
# Pin record and receipts, then the final-image guards. These run against the
# SHIPPED layout, so a bad COPY or a missing runtime package fails the build
# here rather than as a crashloop on the node.
#   gufo-commit        the full pinned SHA (the gate cross-checks it).
#   linked-libs.txt    ldd of the binary with each library's owning package.
#   runtime-packages   dpkg name/version list, for license follow-up.
RUN set -e; \
    mkdir -p /opt/llmkube \
    && printf '%s\n' "${GUFO_SHA}" > /opt/llmkube/gufo-commit \
    && printf '%s %s\n' "${GUFO_REF}" "${GUFO_REPO}" > /opt/llmkube/gufo-ref \
    && ldd /usr/local/bin/gufo > /tmp/ldd.txt 2>&1; \
    if grep -q 'not found' /tmp/ldd.txt; then \
      echo "ERROR: a library gufo links is missing from the runtime stage:"; \
      cat /tmp/ldd.txt; exit 1; \
    fi; \
    if grep -Eiq 'libav(codec|format|util|filter|device)|libsw(scale|resample)|libpostproc' /tmp/ldd.txt; then \
      echo "ERROR: gufo links an FFmpeg library"; exit 1; \
    fi; \
    : > /opt/llmkube/linked-libs.txt; \
    for lib in $(awk '/=> \//{print $3}' /tmp/ldd.txt); do \
      real="$(readlink -f "$lib")"; \
      pkg="$(dpkg -S "$real" 2>/dev/null | head -1 | cut -d: -f1)"; \
      [ -n "$pkg" ] || pkg="$(dpkg -S "$lib" 2>/dev/null | head -1 | cut -d: -f1)"; \
      printf '%s %s\n' "$lib" "${pkg:-UNOWNED}" >> /opt/llmkube/linked-libs.txt; \
    done; \
    dpkg-query -W -f='${Package} ${Version}\n' > /opt/llmkube/runtime-packages.txt; \
    if grep -Eq '^(ffmpeg|libav(codec|format|util|filter|device)[0-9a-z-]*|libsw(scale|resample)[0-9a-z-]*|libpostproc[0-9a-z-]*) ' /opt/llmkube/runtime-packages.txt; then \
      echo "ERROR: an FFmpeg package is installed in the runtime image"; exit 1; \
    fi; \
    if command -v ffmpeg >/dev/null 2>&1 || command -v ffprobe >/dev/null 2>&1; then \
      echo "ERROR: an ffmpeg/ffprobe executable is on PATH"; exit 1; \
    fi; \
    want="gufo version ${GUFO_REF#v} (${GUFO_SHA})"; \
    got="$(/usr/local/bin/gufo --version)"; \
    if [ "$got" != "$want" ]; then echo "ERROR: '$got' != '$want'"; exit 1; fi; \
    /usr/local/bin/gufo --help > /dev/null \
    && echo "PASS: runtime stage resolves every library, carries no FFmpeg, and runs ${want}"
EXPOSE 8080
# Runs as root, like rocm/Dockerfile: ROCm needs /dev/kfd and /dev/dri, and a
# non-root user would need the host's render/video GIDs through
# securityContext.supplementalGroups on every InferenceService. Revisit if this
# image graduates from the bake-off.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1
ENTRYPOINT ["/usr/local/bin/gufo"]
