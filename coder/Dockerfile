# syntax=docker/dockerfile:1
#
# LLMKube Foreman coder-agent toolchain image.
#
# The published foreman-agent image (Dockerfile.foreman-agent in the LLMKube
# repo) is intentionally minimal: git + the binary, no language toolchain. That
# is correct for the reviewer / scheduler roles, but a *coder* agent runs an
# in-workspace self-gate (gofmt, go vet, go build, golangci-lint, changed-package
# go test, codegen-drift) against the target repo, which needs that repo's
# language toolchain present in the agent container. Baking a Go toolchain into
# the published agent image would not generalise (a Python or Rust target repo
# needs a different toolchain), so the coder capability is a per-language BYO
# toolchain image, parallel to AgenticTask.gateProfile.image.
#
# This is LLMKube's own (Go) coder image. Point a coder-role agent at it via the
# charts/foreman values (agent.image + add "coder" to agent.roles). See
# defilantech/LLMKube#835.
#
# Pinned inputs are bumped via reviewed PR (mirrors vulkan/Dockerfile). Nothing
# is compiled under emulation: foreman-agent and controller-gen are
# cross-compiled in the $BUILDPLATFORM build stage, and helm + golangci-lint are
# downloaded as prebuilt per-arch release binaries, so the arm64 image builds as
# fast as amd64.

# foreman-agent source pin: a github.com/defilantech/llmkube module ref the
# `go install` below resolves (verified against the Go checksum database). Pinned
# to the commit carrying the coder-gate convergence trio -- codegen-drift
# auto-resolve (#851), branch-from-upstream (#813), scope-overlap guard (#782) --
# which post-dates the v0.8.19 release. Bump to a release tag (e.g. v0.8.20) once
# those ship in a tagged release.
ARG LLMKUBE_REF=795bf2380168d0351b3259205f6644b59a581bfe

# Toolchain version pins. Keep in lockstep with the LLMKube Makefile
# (HELM_VERSION, GOLANGCI_LINT_VERSION, CONTROLLER_TOOLS_VERSION) so the
# in-cluster fast gate runs the same tool versions CI does.
ARG GO_VERSION=1.26
ARG HELM_VERSION=v3.16.4
ARG GOLANGCI_LINT_VERSION=v2.12.2
ARG CONTROLLER_TOOLS_VERSION=v0.19.0

# Build stage. Pinned to $BUILDPLATFORM (the runner's native arch) and told to
# cross-compile via GOOS/GOARCH, so the Go compiler always runs natively and the
# arm64 leg never pays QEMU's interpreted-compile cost. CGO disabled -> a static
# binary that runs on the glibc runtime base unchanged.
FROM --platform=$BUILDPLATFORM golang:${GO_VERSION} AS build
ARG LLMKUBE_REF
ARG CONTROLLER_TOOLS_VERSION
ARG TARGETOS
ARG TARGETARCH
ENV CGO_ENABLED=0
WORKDIR /build
# Resolve LLMKUBE_REF to the module's version string so the stamped binary
# reports a real version instead of "dev". `go list -m` needs a module context,
# so run it in a throwaway module; for a tag this returns the tag, for a commit
# the proxy's pseudo-version (e.g. v0.8.20-0.<ts>-<sha>). The foreman-agent
# binary reads `main.Version`/`main.GitCommit` (same vars goreleaser stamps), and
# the FleetNode heartbeats main.Version, so an empty stamp shows up as "dev" in
# `kubectl get fleetnodes`.
RUN mkdir -p /out /tmp/vresolve \
    && ( cd /tmp/vresolve && go mod init coderbuild >/dev/null 2>&1 \
         && go list -m -f '{{.Version}}' "github.com/defilantech/llmkube@${LLMKUBE_REF}" > /out/llmkube.version ) \
    && LLMKUBE_VERSION="$(cat /out/llmkube.version)" \
    && echo "foreman-agent version: ${LLMKUBE_VERSION} (ref ${LLMKUBE_REF})"
# go install with GOOS/GOARCH set lands cross-built binaries under
# $GOPATH/bin/<os>_<arch>/ and native ones under $GOPATH/bin/, so resolve each by
# name rather than assuming a path. controller-gen is cross-compiled here too
# (it has no prebuilt release binary) to keep it off the emulated path.
RUN LLMKUBE_VERSION="$(cat /out/llmkube.version)" \
    && GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go install -trimpath \
         -ldflags="-s -w -X main.Version=${LLMKUBE_VERSION} -X main.GitCommit=${LLMKUBE_REF}" \
         github.com/defilantech/llmkube/cmd/foreman-agent@${LLMKUBE_REF} \
    && GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go install \
         sigs.k8s.io/controller-tools/cmd/controller-gen@${CONTROLLER_TOOLS_VERSION} \
    && cp "$(find "$(go env GOPATH)/bin" -name foreman-agent -type f | head -1)" /out/foreman-agent \
    && cp "$(find "$(go env GOPATH)/bin" -name controller-gen -type f | head -1)" /out/controller-gen \
    && test -x /out/foreman-agent && test -x /out/controller-gen

# Runtime stage: golang:1.26 -- the same base the gate Job uses, so the coder's
# `go build` / `go test` see a real Go toolchain (go + git ship in the image).
# Add make + helm + golangci-lint + controller-gen so every fast-gate check
# resolves without spawning a separate gate Job.
FROM golang:${GO_VERSION}
ARG TARGETARCH
ARG HELM_VERSION
ARG GOLANGCI_LINT_VERSION

LABEL org.opencontainers.image.title="llmkube-foreman-agent-coder" \
      org.opencontainers.image.description="LLMKube Foreman coder-agent image: foreman-agent + Go toolchain (make, helm, golangci-lint, controller-gen) for the in-workspace coder self-gate" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0"

# make is the only gate dependency the golang image lacks (go + git ship in it).
# curl + ca-certs fetch helm and golangci-lint.
RUN apt-get update \
    && apt-get install -y --no-install-recommends make curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# helm: coder tasks that edit charts verify with `helm template` / `helm lint`.
# Prebuilt per-arch tarball -- no compile. Pinned for reproducibility.
RUN curl -fsSL "https://get.helm.sh/helm-${HELM_VERSION}-linux-${TARGETARCH:-amd64}.tar.gz" \
        | tar -xz -C /tmp \
    && mv "/tmp/linux-${TARGETARCH:-amd64}/helm" /usr/local/bin/helm \
    && chmod 0755 /usr/local/bin/helm \
    && rm -rf /tmp/linux-* \
    && helm version --short

# golangci-lint: the fast gate runs `golangci-lint run`. Prebuilt per-arch
# release tarball (the v2 binary is heavy to compile, so download it rather than
# `go install` under emulation). Pinned to the Makefile's GOLANGCI_LINT_VERSION.
RUN ver="${GOLANGCI_LINT_VERSION#v}" \
    && dir="golangci-lint-${ver}-linux-${TARGETARCH:-amd64}" \
    && curl -fsSL "https://github.com/golangci/golangci-lint/releases/download/${GOLANGCI_LINT_VERSION}/${dir}.tar.gz" \
        | tar -xz -C /tmp \
    && mv "/tmp/${dir}/golangci-lint" /usr/local/bin/golangci-lint \
    && chmod 0755 /usr/local/bin/golangci-lint \
    && rm -rf "/tmp/golangci-lint-"* \
    && golangci-lint version

# The coder runs non-root (charts/foreman podSecurityContext runAsNonRoot:true)
# with a read-only root filesystem; only /workspace, /tmp and /cache are
# writable. Bake git's safe.directory into /etc/gitconfig now (build time) --
# at runtime /etc is read-only, and without it git refuses to operate on the
# cloned workspace, which a different uid owns.
RUN git config --system --add safe.directory '*'

# Binary at /foreman-agent: the charts/foreman agent Deployment invokes
# `command: ["/foreman-agent"]` (absolute path); the gate Job template invokes it
# by bare name off PATH. /usr/local/bin is on PATH, so symlink both names to one
# binary. controller-gen lands beside it for the codegen-drift check.
COPY --from=build /out/foreman-agent /foreman-agent
COPY --from=build /out/controller-gen /usr/local/bin/controller-gen
RUN chmod 0755 /foreman-agent /usr/local/bin/controller-gen \
    && ln -s /foreman-agent /usr/local/bin/foreman-agent

# HOME, GOPATH and GOCACHE must resolve to writable paths: the root FS is
# read-only and the base image hard-codes GOPATH=/go (read-only here), so point
# all three at the writable /tmp emptyDir. Without this `go build` cannot write
# its module/build cache and the gate fails to start.
ENV HOME=/tmp \
    GOPATH=/tmp/go \
    GOCACHE=/tmp/.cache/go-build

# Numeric non-root user (no /etc/passwd entry needed). 65532 = the distroless
# "nonroot" uid; it satisfies the runAsNonRoot:true pod policy with no explicit
# runAsUser. Go + git work fine with a bare uid as long as HOME is writable.
USER 65532:65532

# command: ["/foreman-agent"] in the Deployment overrides this; set it so the
# image is runnable and self-documenting standalone.
ENTRYPOINT ["/foreman-agent"]
