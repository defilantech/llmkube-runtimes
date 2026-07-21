# rocm-smi-exporter

Small Prometheus exporter for AMD GPU / ROCm telemetry on Kubernetes nodes.

It reads the Linux **AMDGPU sysfs + hwmon** files directly — no `rocm-smi` binary
and no ROCm userspace in the image. That keeps the container rootless, static,
multi-arch, and Talos-friendly, and — critically — it **enumerates the Strix Halo
`gfx1151` iGPU**, which the Instinct-scoped `device-metrics-exporter` /
`amd_smi_exporter` do not (see [defilantech/LLMKube#700](https://github.com/defilantech/LLMKube/issues/700)).

- **Source:** `cmd/rocm-smi-exporter` (pure Go stdlib, no module dependencies).
- **Image:** `ghcr.io/defilantech/llmkube-rocm-smi-exporter` (distroless `nonroot`, multi-arch).
- **Endpoint:** `GET /metrics` on `:9494`.

## Configuration

| Variable      | Default | Description                                    |
| ------------- | ------- | ---------------------------------------------- |
| `LISTEN_ADDR` | `:9494` | HTTP listen address                            |
| `SYSFS_ROOT`  | `/sys`  | Root of the sysfs tree to scrape               |

In Kubernetes, mount the host `/sys` read-only and set `SYSFS_ROOT=/host/sys`.

## Metrics

Discovery/health: `rocm_smi_gpus_discovered`, `rocm_smi_gpu_info{card,pci_slot,vendor_id,device_id,…}`, `rocm_smi_scrape_success`, `rocm_smi_scrape_failures_total`, `rocm_smi_last_scrape_duration_seconds`.

AMDGPU device (when the kernel exposes them): `rocm_smi_gpu_busy_percent`, `rocm_smi_memory_busy_percent`, `rocm_smi_vram_used_bytes`, `rocm_smi_vram_total_bytes`, `rocm_smi_visible_vram_{used,total}_bytes`, `rocm_smi_gtt_{used,total}_bytes`, `rocm_smi_pcie_replay_total (counter)`.

HWMON (when exposed): `rocm_smi_temperature_celsius{sensor}`, `rocm_smi_power_watts{type}`, `rocm_smi_fan_rpm{sensor}`, `rocm_smi_clock_hertz{sensor}`, `rocm_smi_voltage_volts{sensor}`.

## DaemonSet (reference)

Deploy on AMD nodes, mounting host `/sys` read-only. The exporter writes nothing,
so it runs read-only-rootfs as `nobody`:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: rocm-smi-exporter
spec:
  selector:
    matchLabels: { app.kubernetes.io/name: rocm-smi-exporter }
  template:
    metadata:
      labels: { app.kubernetes.io/name: rocm-smi-exporter }
    spec:
      nodeSelector:
        node-role.kubernetes.io/rocm-worker: "true"
      containers:
        - name: exporter
          image: ghcr.io/defilantech/llmkube-rocm-smi-exporter:latest
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
docker build -f rocm-smi-exporter/Dockerfile -t rocm-smi-exporter:dev .
go test ./cmd/rocm-smi-exporter/...
```
