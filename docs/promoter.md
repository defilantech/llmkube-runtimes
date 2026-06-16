# Image promoter (Tier-2 gate)

The promoter turns CI-built **candidate** images into trusted **stable** images
by verifying provenance and proving real GPU inference on hardware. It runs
out-of-band on a self-hosted gfx1151 host, never as a CI runner, so untrusted
fork-PR code never touches the GPU box.

## Flow

```
ghcr :candidate-<sha>  ──►  promoter (systemd timer on the GPU host)
                              1. cosign verify build-provenance (fail-closed)
                              2. sandboxed offline GPU smoke Job (kubectl)
                                 - Vulkan0 device present
                                 - layers offloaded to GPU
                                 - decode tokens/sec >= floor
                              3. on pass: retag digest -> :stable + :b<ref>-llmkube<N>
```

A candidate that fails provenance is recorded and never smoked or promoted. A
candidate that loads but falls back to CPU (no `Vulkan0`) fails the smoke and is
not promoted. State (processed digests + verdicts) lives in a JSON file so each
run only processes new candidates.

## Two-tier gate, end to end

- **Tier 1 (CI, free runners, no GPU):** build + an `RTLD_NOW` dlopen of
  `libggml-vulkan.so` (catches the [#725](https://github.com/defilantech/LLMKube/issues/725)-class
  undefined-symbol break) + a runtime launch check. Publishes `:candidate-<sha>`
  with build provenance.
- **Tier 2 (this promoter, real gfx1151):** provenance verify + GPU smoke +
  promote. v1 is verify-only; a smoke-passed cosign signature is a fast-follow.

## The sandbox

The smoke runs as a Kubernetes Job (`vulkan/smoke/`) that is:

- non-root, read-only rootfs, all capabilities dropped, `automountServiceAccountToken: false`;
- offline: a `deny-egress` NetworkPolicy, model pre-staged read-only (no network);
- device-scoped: requests `devic.es/dri-render` (the generic device-plugin
  resource) and runs with `supplementalGroups: [<render gid>]` for `/dev/dri`;
- ephemeral: `activeDeadlineSeconds` + `ttlSecondsAfterFinished`, `backoffLimit: 0`.

The promoter process holds the ghcr push credential and the kubeconfig; the
smoke Job sees neither.

## Run it

```bash
go build -o promoter ./cmd/promoter
./promoter run-once --render-gid "$(getent group render | cut -d: -f3)"
```

Flags: `--repo` (default `ghcr.io/defilantech/llmkube-llama-vulkan`),
`--attest-repo` (the repo whose CI signed the provenance),
`--min-decode-toks` (smoke throughput floor), `--namespace`, `--state`.

## Deploy on the GPU host

1. Apply the cluster pieces: `kubectl apply -f vulkan/smoke/namespace.yaml -f vulkan/smoke/networkpolicy.yaml -f vulkan/smoke/rbac.yaml`.
2. Pre-stage a small smoke model at `/var/lib/llmkube-promoter/models/smoke.gguf`.
3. Install the binary at `~/.local/bin/promoter`, the systemd units from
   `deployment/shadowstrix/`, and `~/.config/llmkube-promoter/env` (see
   `llmkube-promoter.env.example`).
4. `systemctl --user enable --now llmkube-promoter.timer`.

Host-specific values (render GID, token scopes, kubeconfig) are maintainer ops,
not committed here.

## Bumping the image

Edit `LLAMACPP_REF` + `LLAMACPP_SHA` in `vulkan/Dockerfile` (the build fails if
they disagree). CI builds a new candidate; the promoter validates and promotes
it. Adoption is release-driven; a weekly CI canary is an early-warning build.
