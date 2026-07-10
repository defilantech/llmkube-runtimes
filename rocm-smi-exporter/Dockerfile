# syntax=docker/dockerfile:1
#
# rocm-smi-exporter: a small Prometheus exporter for AMD GPU telemetry on
# Kubernetes nodes. It reads the Linux AMDGPU sysfs + hwmon trees directly, so it
# needs no rocm-smi binary and no ROCm userspace in the image — which is exactly
# what lets it enumerate the Strix Halo gfx1151 iGPU that the Instinct-scoped
# device-metrics-exporter / amd_smi_exporter cannot see (defilantech/LLMKube#700).
#
# The binary is pure Go stdlib (no module dependencies) with CGO disabled, so it
# is a fully static binary on the distroless nonroot base and cross-compiles for
# arm64 in the $BUILDPLATFORM stage without paying QEMU's interpreted-compile
# cost — mirroring coder/Dockerfile and vulkan/Dockerfile.

ARG GO_VERSION=1.26

# Build stage. Pinned to $BUILDPLATFORM (the runner's native arch) and told to
# cross-compile via GOOS/GOARCH, so the Go compiler always runs natively. CGO
# disabled -> a static binary that runs on the distroless base unchanged.
FROM --platform=$BUILDPLATFORM golang:${GO_VERSION} AS build
ARG TARGETOS
ARG TARGETARCH
# VERSION stamps main.version (surfaced in logs / build info). A tag push sets it
# from the ref; branch/PR/cron builds leave the "dev" default.
ARG VERSION=dev
ENV CGO_ENABLED=0
WORKDIR /src
# stdlib-only: go.mod/go.sum give the module context; the package pulls no
# external deps, so nothing is downloaded.
COPY go.mod go.sum ./
COPY cmd/rocm-smi-exporter ./cmd/rocm-smi-exporter
RUN GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go build -trimpath \
        -ldflags="-s -w -X main.version=${VERSION}" \
        -o /out/rocm-smi-exporter ./cmd/rocm-smi-exporter \
    && test -x /out/rocm-smi-exporter

# Runtime: distroless static nonroot — no shell, no package manager, uid 65534.
# The exporter mounts the host /sys read-only (SYSFS_ROOT=/host/sys) and serves
# /metrics on :9494. Read-only-rootfs friendly: it writes nothing.
FROM gcr.io/distroless/static-debian13:nonroot

LABEL org.opencontainers.image.title="llmkube-rocm-smi-exporter" \
      org.opencontainers.image.description="Prometheus exporter for AMD GPU telemetry via AMDGPU sysfs/hwmon; enumerates the gfx1151 iGPU without a ROCm userspace" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=build /out/rocm-smi-exporter /rocm-smi-exporter

USER 65534:65534
EXPOSE 9494
ENTRYPOINT ["/rocm-smi-exporter"]
