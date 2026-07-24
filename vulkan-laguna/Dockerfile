# syntax=docker/dockerfile:1
#
# LLMKube AMD/Vulkan llama.cpp server runtime, Laguna variant.
#
# This is a VARIANT image, not a replacement for vulkan/Dockerfile. It exists
# for exactly one reason: serving poolside Laguna models with working tool
# calling. Upstream ggml-org llama.cpp can LOAD Laguna (the arch merged
# 2026-07-22, first release b10103) but cannot PARSE its tool calls: Laguna
# emits poolside's native format and upstream's common/chat.cpp has no handler
# for it, so a tool call arrives as raw text in message.content with
# finish_reason "stop". Anything driving an agent loop over the OpenAI API
# (LLMKube's Foreman coder, for one) sees no tool call at all.
#
# TheTom's llama.cpp fork solves this. See docs/laguna-variant.md for the full
# story, the provenance, and the serving contract.
#
# The production image (vulkan/Dockerfile) stays PURE upstream and is not
# affected by this file. Consume this variant only for Laguna-family models.

# TheTom's fork, branch laguna/port, pinned by immutable commit SHA.
#
# The pin is a bare SHA with no tag or branch ref on purpose: laguna/port is an
# active development branch whose HEAD moves, and a --branch clone would
# silently build different source over time. The build fetches this exact
# commit (GitHub serves reachable SHAs), so a bump is always a reviewed edit
# here. Bump by editing LLAMACPP_SHA; there is no companion ref to keep in sync.
ARG LLAMACPP_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
ARG LLAMACPP_SHA=3e7bbe1ead3fa37963ba1cc82771eea72627ebb3
# ubuntu:26.04 pinned by digest, same base as the production Vulkan image:
# 26.04's Mesa is new enough for gfx1151 (Strix Halo) RADV.
ARG BASE_DIGEST=sha256:f3d28607ddd78734bb7f71f117f3c6706c666b8b76cbff7c9ff6e5718d46ff64

FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS build
ARG LLAMACPP_REPO
ARG LLAMACPP_SHA
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ca-certificates \
      libvulkan-dev glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Fetch the pinned commit directly. --depth 1 on a SHA keeps the clone small
# without pulling branch history, and checking out FETCH_HEAD means the build
# fails loudly if the SHA ever stops being served rather than drifting to a
# branch tip.
RUN git init -q . \
    && git remote add origin "${LLAMACPP_REPO}" \
    && git fetch -q --depth 1 origin "${LLAMACPP_SHA}" \
    && git checkout -q FETCH_HEAD \
    && got="$(git rev-parse HEAD)" \
    && if [ "${got}" != "${LLAMACPP_SHA}" ]; then \
         echo "ERROR: checked out ${got}, expected ${LLAMACPP_SHA}"; exit 1; \
       fi
# Identical flags to the production Vulkan image so this variant differs from it
# in source pin ONLY, never in build configuration. See vulkan/Dockerfile for
# why each one is set (dynamic backend loading, one generic CPU backend, no
# curl, and the prebuilt web UI disabled because a --depth 1 clone has no tag
# history for the UI's git-describe build number).
RUN cmake -B build \
      -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out /out-templates \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-server /out/ \
    && cp models/templates/poolside-Laguna-*.jinja /out-templates/
# Laguna arch guard, GPU-independent. Same guard as the production image: the
# arch-name string compiles into libllama (not llama-server), so grep /out
# recursively. A pin predating laguna support would build clean but silently
# fail to load Laguna GGUFs.
RUN grep -rqa laguna /out/ \
    || { echo "ERROR: laguna arch not present in the built llama.cpp"; exit 1; }
# Tool-call parser guard, GPU-independent. THE reason this variant exists.
# TheTom's fork parses tool calls with a "differential autoparser": it renders
# the jinja chat template with varying inputs, diffs the outputs to derive the
# tool-call delimiters, and generates a parser. Pure upstream has no such code.
# This asserts we built the fork and not, say, an upstream commit someone
# retargeted LLAMACPP_REPO at, which would produce an image that serves Laguna
# perfectly while silently emitting zero tool calls.
RUN grep -rqa "differential autoparser" /out/ \
    || { echo "ERROR: fork autoparser absent; LLAMACPP_SHA is not TheTom laguna/port"; exit 1; }
# Chat-template guard, GPU-independent. The autoparser learns the tool-call
# format FROM the template, so a template that omits the argument delimiters
# yields a parser that silently drops every tool call. poolside's own released
# GGUF embeds exactly such a template (it has <tool_call> but no arg_key), which
# is why this image ships the fork's corrected templates and defaults to one.
# Assert the shipped template carries the argument encoding before we build an
# image whose entire purpose depends on it.
RUN grep -qa 'arg_key' /out-templates/poolside-Laguna-S-2.1.jinja \
    || { echo "ERROR: shipped chat template lacks arg_key; tool calls would not parse"; exit 1; }
# #725 guard, GPU-independent. The upstream break was an undefined shader symbol
# in libggml-vulkan.so that bound lazily, so it only crashed at shader-use time
# on a real GPU. RTLD_NOW forces every relocation to resolve at dlopen, so a
# build with that defect fails here instead of shipping.
RUN printf '%s\n' \
      '#include <dlfcn.h>' '#include <stdio.h>' \
      'int main(int argc, char **argv) {' \
      '  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);' \
      '  if (!h) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }' \
      '  printf("dlopen OK: %s\n", argv[1]); return 0; }' > /tmp/dlcheck.c \
    && gcc /tmp/dlcheck.c -o /tmp/dlcheck -ldl \
    && LD_LIBRARY_PATH=/out /tmp/dlcheck /out/libggml-vulkan.so

FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS server
ARG LLAMACPP_REPO
ARG LLAMACPP_SHA
LABEL org.opencontainers.image.title="llmkube-llama-vulkan-laguna" \
      org.opencontainers.image.description="LLMKube AMD/Vulkan llama.cpp server runtime, Laguna variant (poolside tool calling)" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.llamacpp.repo="${LLAMACPP_REPO}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}" \
      io.llmkube.variant="laguna"
# RADV (mesa-vulkan-drivers) + the Vulkan loader. curl is only for the
# container HEALTHCHECK. No ROCm.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libvulkan1 mesa-vulkan-drivers curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r llama && useradd -r -g llama llama
COPY --from=build /out/ /app/
COPY --from=build /out-templates/ /app/templates/
WORKDIR /app
# GGML_BACKEND_DL looks beside the executable; /app holds llama-server and the
# libggml-*.so backends.
ENV LLAMA_ARG_HOST=0.0.0.0
# Default the chat template to the fork's corrected Laguna-S-2.1 one, because
# the model's OWN embedded template does not parse tool calls (see the template
# guard above). Defaulting it here makes the image correct out of the box
# instead of depending on every caller remembering a flag. llama.cpp applies env
# defaults before command-line args, so serving another Laguna (for example an
# XS variant) is still just --chat-template-file /app/templates/<other>.jinja.
ENV LLAMA_ARG_CHAT_TEMPLATE_FILE=/app/templates/poolside-Laguna-S-2.1.jinja
EXPOSE 8080
# Non-root. GPU access needs the deployment to grant the host render group via
# securityContext.supplementalGroups (the device plugin exposes /dev/dri).
USER llama
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1
ENTRYPOINT ["/app/llama-server"]
