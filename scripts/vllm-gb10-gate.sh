#!/usr/bin/env bash
#
# Tier-1 runtime gate for the GB10 vLLM NVFP4 image (cheap, no GPU required).
#
# This image is not compiled here: it is a pinned third-party base plus one pip
# layer. So the gate asks different questions than cuda-gate.sh, which verifies
# device code we emitted ourselves. Here the risks are:
#
#   1. the b12x upgrade silently did not apply (wrong site-packages, a base that
#      pins b12x, pip resolving to a cached older wheel), leaving the image with
#      exactly the bugs it exists to fix;
#   2. `pip install --no-deps` satisfied nothing and left the environment
#      internally inconsistent;
#   3. the base moved and no longer has the layout the Dockerfile guards assume.
#
# Everything below runs against the SHIPPED image rather than the build stage.
# The Dockerfile's own guards ran during build; this proves they describe what
# was actually published, which is the check that survives a cache hit.
#
# NOTHING HERE IMPORTS b12x OR vllm. Both initialize CUDA on import and there is
# no device on a hosted runner; importing them would fail for a reason that has
# nothing to do with the image being correct. Files and metadata only.
set -euo pipefail

IMAGE="${1:?usage: vllm-gb10-gate.sh <image-ref>}"
WANT_B12X="${2:?usage: vllm-gb10-gate.sh <image-ref> <expected-b12x-version>}"

run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

# --- 1. The b12x upgrade landed ----------------------------------------------
echo "== b12x version in the shipped image =="
if ! got="$(run python3 -c 'import importlib.metadata as m; print(m.version("b12x"))' 2>&1)"; then
  echo "${got}"
  echo "FAIL: could not read b12x version from the image"
  exit 1
fi
got="$(printf '%s' "${got}" | tr -d '[:space:]')"
echo "b12x == ${got} (expected ${WANT_B12X})"
if [ "${got}" != "${WANT_B12X}" ]; then
  echo "FAIL: image ships b12x ${got}, Dockerfile pins ${WANT_B12X}."
  echo "      A base that pins b12x itself would produce exactly this."
  exit 1
fi
echo "PASS: b12x ${WANT_B12X} is the version the image actually carries"

# --- 2. The two W4A16 bugs are absent from the shipped files ------------------
#
# The whole reason this image exists. Asserted on file contents, not on the
# version string above: a version number is a claim, the source is the fact.
echo "== W4A16 CUDA-graph guards =="
# Passed with `python3 -c "$PROBE"`, never on stdin: `docker run` without -i
# does not forward stdin, so a heredoc here would hand python an EMPTY program.
# That exits 0 and the gate would report PASS having checked nothing -- the
# quietest possible way for a guard to stop guarding.
PROBE='
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("b12x")
if spec is None or not spec.origin:
    sys.exit("b12x not found in the image")
root = pathlib.Path(spec.origin).parent / "moe/_shared/kernels/w4a16"
if not root.is_dir():
    sys.exit("w4a16 kernel directory missing at %s" % root)
if "metadata_row" in (root / "kernel.py").read_text(errors="replace"):
    sys.exit("kernel.py still references metadata_row (output-drain bug present)")
if "is_current_stream_capturing" not in (root / "route_pack.py").read_text(errors="replace"):
    sys.exit("route_pack.py has no capture guard (workspace still allocated per call)")
print("no metadata_row; capture guard present")
'
if ! out="$(run python3 -c "${PROBE}" 2>&1)"; then
  echo "${out}"
  echo "FAIL: the shipped b12x still carries a W4A16 CUDA-graph defect."
  echo "      Serving would start and then crash during graph capture with MTP on."
  exit 1
fi
echo "${out}"
echo "PASS: both W4A16 defects absent from the shipped b12x"

# --- 3. --no-deps left a coherent environment ---------------------------------
#
# `pip install --no-deps` is deliberate (see the Dockerfile) but it is exactly
# the flag that can leave an unsatisfiable requirement behind if a future b12x
# grows a dependency the base does not have. pip check is the cheap way to
# notice that at build time instead of at import time on a GB10.
echo "== pip check =="
if ! out="$(run python3 -m pip check 2>&1)"; then
  echo "${out}"
  echo "FAIL: broken dependencies in the image after the --no-deps b12x install."
  echo "      If b12x gained a real dependency, install it explicitly rather"
  echo "      than dropping --no-deps, which would let torch/vllm be replaced."
  exit 1
fi
echo "${out}"
echo "PASS: dependency set is coherent"

# --- 4. The server entrypoint still exists ------------------------------------
#
# `command -v`, not `vllm --version`: the latter imports vllm, which initializes
# CUDA. This only asks whether our pip layer clobbered the console script.
echo "== vllm entrypoint =="
if ! out="$(run sh -lc 'command -v vllm' 2>&1)"; then
  echo "${out}"
  echo "FAIL: no vllm executable on PATH in the shipped image"
  exit 1
fi
echo "${out}"
echo "PASS: vllm entrypoint present"

echo "== verdict =="
echo "PASS: GB10 vLLM NVFP4 image gated (b12x ${WANT_B12X}, W4A16 clean, deps coherent)"
