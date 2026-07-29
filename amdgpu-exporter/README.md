# amdgpu-exporter

Small Prometheus exporter for AMD GPU / ROCm telemetry on Kubernetes nodes.

It reads the Linux **AMDGPU sysfs + hwmon** files directly — no `rocm-smi` binary
and no ROCm userspace in the image. That keeps the container rootless, static,
multi-arch, and Talos-friendly, and — critically — it **enumerates the Strix Halo
`gfx1151` iGPU**, which the Instinct-scoped `device-metrics-exporter` /
`amd_smi_exporter` do not (see [defilantech/LLMKube#700](https://github.com/defilantech/LLMKube/issues/700)).

- **Source:** `cmd/amdgpu-exporter` (pure Go stdlib, no module dependencies).
- **Image:** `ghcr.io/defilantech/llmkube-amdgpu-exporter` (distroless `nonroot`, multi-arch).
- **Endpoint:** `GET /metrics` on `:9494`.

## Configuration

| Variable      | Default | Description                                    |
| ------------- | ------- | ---------------------------------------------- |
| `LISTEN_ADDR` | `:9494` | HTTP listen address                            |
| `SYSFS_ROOT`  | `/sys`  | Root of the sysfs tree to scrape               |

In Kubernetes, mount the host `/sys` read-only and set `SYSFS_ROOT=/host/sys`.

## Metrics

Discovery/health: `amdgpu_gpus_discovered`, `amdgpu_gpu_info{card,pci_slot,vendor_id,device_id,…}`, `amdgpu_scrape_success`, `amdgpu_scrape_failures_total`, `amdgpu_last_scrape_duration_seconds`.

AMDGPU device (when the kernel exposes them): `amdgpu_gpu_busy_percent`, `amdgpu_memory_busy_percent`, `amdgpu_vram_used_bytes`, `amdgpu_vram_total_bytes`, `amdgpu_visible_vram_{used,total}_bytes`, `amdgpu_gtt_{used,total}_bytes`, `amdgpu_pcie_replay_total (counter)`.

HWMON (when exposed): `amdgpu_temperature_celsius{sensor}`, `amdgpu_power_watts{type}`, `amdgpu_fan_rpm{sensor}`, `amdgpu_clock_hertz{sensor}`, `amdgpu_voltage_volts{sensor}`.

## DaemonSet (reference)

Deploy on AMD nodes, mounting host `/sys` read-only. The exporter writes nothing,
so it runs read-only-rootfs as `nobody`:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: amdgpu-exporter
spec:
  selector:
    matchLabels: { app.kubernetes.io/name: amdgpu-exporter }
  template:
    metadata:
      labels: { app.kubernetes.io/name: amdgpu-exporter }
    spec:
      nodeSelector:
        node-role.kubernetes.io/rocm-worker: "true"
      containers:
        - name: exporter
          image: ghcr.io/defilantech/llmkube-amdgpu-exporter:latest
          env:
            - { name: SYSFS_ROOT, value: /host/sys }
          ports:
            - { name: metrics, containerPort: 9494 }
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: { drop: ["ALL"] }
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 65534
            runAsGroup: 65534
          volumeMounts:
            - { name: sys, mountPath: /host/sys, readOnly: true }
      volumes:
        - name: sys
          hostPath: { path: /sys, type: Directory }
```

## Strix Halo notes

Start bottleneck work from GPU busy, memory busy, VRAM/GTT allocation, clocks,
power, and temperature. AMDGPU sysfs exposes memory utilization as a busy
percentage, not always true bandwidth (GB/s). If `gpu_metrics` / AMD SMI CPU
metrics expose richer values on a given `gfx1151` box, add them after validating
the files present on the node.

## Build

```bash
# from the repo root (context must be the module root)
docker build -f amdgpu-exporter/Dockerfile -t amdgpu-exporter:dev .
go test ./cmd/amdgpu-exporter/...
```
