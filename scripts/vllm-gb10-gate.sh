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

# --- 3. The layer introduced no NEW dependency conflict ------------------------
#
# NOT a bare `pip check`. The base is already internally inconsistent: it ships
# nvidia-cutlass-dsl 4.7.0 while b12x (at every version, the pin has not moved)
# and quack-kernels 0.6.4 both require exactly 4.6.2. quack-kernels is a base
# package this image never touches, which is what proves the conflict is
# inherited rather than caused by our --no-deps install.
#
# A bare pip check therefore fails on a condition we did not create and cannot
# responsibly "fix" -- pinning cutlass-dsl down to 4.6.2 would change a CUDA DSL
# out from under a vLLM build that works today. But dropping the check entirely
# would give up the one guard that catches a future b12x bump quietly adding a
# dependency.
#
# So: diff against the baseline the Dockerfile recorded BEFORE the install.
# Inherited conflicts are reported and pass; anything NEW fails.
echo "== dependency conflicts introduced by this layer =="

BASELINE_PATH=/opt/llmkube/pip-check-baseline.txt
if ! base_raw="$(run cat "${BASELINE_PATH}" 2>&1)"; then
  echo "${base_raw}"
  echo "FAIL: ${BASELINE_PATH} missing from the image."
  echo "      The Dockerfile records it before installing b12x; without it this"
  echo "      check cannot tell an inherited conflict from one we introduced."
  exit 1
fi

if ! live_raw="$(run python3 -m pip check 2>&1)"; then
  : # non-zero is expected while the inherited conflict stands; classify below.
fi

# Normalize away the SUBJECT package's own version, so upgrading b12x 1.2.6 ->
# 1.3.0 does not make its pre-existing complaint look like a brand new one.
# "b12x 1.2.6 has requirement X, but you have Y." and
# "b12x 1.3.0 has requirement X, but you have Y." both reduce to
# "b12x has requirement X, but you have Y."
normalize() { sed -E 's/^([A-Za-z0-9._-]+) [0-9][^ ]* has requirement /\1 has requirement /' | sed '/^[[:space:]]*$/d' | sort -u; }

base_norm="$(printf '%s\n' "${base_raw}" | normalize)"
live_norm="$(printf '%s\n' "${live_raw}" | normalize)"

if [ -n "${base_norm}" ]; then
  echo "-- inherited from the base (not introduced here, not failing the gate):"
  printf '%s\n' "${base_norm}" | sed 's/^/     /'
fi

# comm needs sorted input; normalize already sorts. -13 = lines only in live.
introduced="$(comm -13 <(printf '%s\n' "${base_norm}") <(printf '%s\n' "${live_norm}") || true)"
introduced="$(printf '%s\n' "${introduced}" | sed '/^[[:space:]]*$/d')"

if [ -n "${introduced}" ]; then
  echo "-- INTRODUCED by the b12x layer:"
  printf '%s\n' "${introduced}" | sed 's/^/     /'
  echo "FAIL: this layer added a dependency conflict that the base did not have."
  echo "      b12x most likely gained or moved a requirement. Install it"
  echo "      explicitly rather than dropping --no-deps, which would let pip"
  echo "      replace torch or vllm underneath a CUDA stack tuned for sm_121."
  exit 1
fi
echo "PASS: no dependency conflict introduced by this layer"

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
