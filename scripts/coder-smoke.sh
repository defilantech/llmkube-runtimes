#!/usr/bin/env bash
#
# Tier-1 smoke for the coder toolchain image. The point of this image is that a
# *non-root* coder agent can run its in-workspace self-gate, so the smoke runs
# the image's default user and execs every tool that gate shells out to. A
# broken toolchain install, a wrong-arch download, a PATH regression, or a
# non-writable HOME fails here before the image is ever trusted. No GPU and no
# cluster needed -- this is the full gate for the coder image.
set -euo pipefail

IMG="${1:?usage: coder-smoke.sh <image>}"

# --entrypoint sh runs as the image's default USER (65532), the same identity
# the Deployment runs the agent under, so a "works as root but not as 65532"
# regression is caught.
docker run --rm --entrypoint sh "${IMG}" -ec '
  echo "== identity ==";          id
  echo "== go ==";                go version
  echo "== make ==";              make --version | head -1
  echo "== git ==";               git --version
  echo "== helm ==";              helm version --short
  echo "== golangci-lint ==";     golangci-lint version
  echo "== controller-gen ==";    controller-gen --version
  echo "== foreman-agent path =="; command -v foreman-agent
  echo "== /foreman-agent ==";    test -x /foreman-agent && echo "/foreman-agent is executable"
  echo "== foreman-agent version (must be stamped, not dev) =="
  v="$(/foreman-agent --version 2>&1)"; echo "$v"
  case "$v" in *" dev"|*"version dev"*) echo "FAIL: version not stamped (got dev)"; exit 1;; esac
  echo "== writable HOME ==";     touch "${HOME}/.coder-smoke" && echo "HOME=${HOME} writable"
  echo "== writable GOCACHE ==";  go env GOCACHE GOPATH
'

echo "coder-smoke: PASS for ${IMG}"
