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

# --- 3. No UNEXPECTED dependency conflict -------------------------------------
#
# Not a bare `pip check`, and not a naive baseline diff either. Both were tried
# and both were wrong, for reasons worth recording so they are not retried:
#
#   - Bare pip check fails, because the image carries nvidia-cutlass-dsl 4.7.0
#     while b12x pins ==4.6.2. That looks like our --no-deps install misfired.
#   - A plain baseline diff also fails, because the BASE has no pip-visible b12x
#     requirements at all (its baseline names only quack-kernels). So every b12x
#     requirement is technically "new" the moment we install the wheel, even
#     though we changed no dependency's version.
#
# What actually changed is VISIBILITY, not content: cutlass-dsl is 4.7.0 before
# and after. And the base maintainer picked 4.7.0 while quack-kernels 0.6.4 --
# a base package we never touch -- also asks for 4.6.2, so 4.7.0 is a deliberate
# base choice made over an existing pin, not an accident we introduced.
#
# So the cutlass-dsl family is allowlisted, LOUDLY and by name, and everything
# else must be either inherited or absent. A future b12x that needs, say,
# flash-attn still fails here, which is the case --no-deps genuinely makes risky.
#
# The allowlist is a statement about metadata, NOT a claim that the kernels work.
# Whether b12x 1.3.0's W4A16 path runs against cutlass-dsl 4.7.0 is a GPU
# question this gate cannot answer; the GB10 smoke test settles it.
echo "== what the base actually shipped =="
if ! receipt="$(run cat /opt/llmkube/base-receipt.txt 2>&1)"; then
  echo "${receipt}"
  echo "FAIL: /opt/llmkube/base-receipt.txt missing; the Dockerfile writes it"
  echo "      before installing b12x. Without it this check is guessing."
  exit 1
fi
printf '%s\n' "${receipt}" | sed 's/^/   /'

echo "== dependency conflicts =="
base_raw="$(printf '%s\n' "${receipt}" | sed -n '/^### pip check/,/^### versions/p' | sed '/^###/d')"
if ! live_raw="$(run python3 -m pip check 2>&1)"; then
  : # non-zero expected while the cutlass-dsl mismatch stands; classified below.
fi

# Drop the subject package's own version so a b12x bump is not mistaken for a
# new conflict: "b12x 1.3.0 has requirement X" -> "b12x has requirement X".
normalize() { sed -E 's/^([A-Za-z0-9._-]+) [0-9][^ ]* has requirement /\1 has requirement /' | sed '/^[[:space:]]*$/d' | sort -u; }
# The one documented mismatch: anything whose UNSATISFIED requirement is in the
# nvidia-cutlass-dsl family. Matches the requirement target, not the subject, so
# a b12x conflict over some other package is never swallowed by this.
CUTLASS_FAMILY='has requirement nvidia-cutlass-dsl(-libs-(base|core|cu12|cu13))?=='

base_norm="$(printf '%s\n' "${base_raw}" | normalize)"
live_norm="$(printf '%s\n' "${live_raw}" | normalize)"

allowed="$(printf '%s\n' "${live_norm}" | grep -E "${CUTLASS_FAMILY}" || true)"
if [ -n "${allowed}" ]; then
  echo "-- KNOWN cutlass-dsl metadata mismatch (allowlisted; NOT a claim it runs):"
  printf '%s\n' "${allowed}" | sed 's/^/     /'
fi

# Everything that is neither allowlisted nor already present in the base.
live_rest="$(printf '%s\n' "${live_norm}" | grep -Ev "${CUTLASS_FAMILY}" || true)"
unexpected="$(comm -13 <(printf '%s\n' "${base_norm}") <(printf '%s\n' "${live_rest}" | sort -u) || true)"
unexpected="$(printf '%s\n' "${unexpected}" | sed '/^[[:space:]]*$/d')"

if [ -n "${unexpected}" ]; then
  echo "-- UNEXPECTED, neither inherited nor allowlisted:"
  printf '%s\n' "${unexpected}" | sed 's/^/     /'
  echo "FAIL: a dependency conflict appeared that this image does not account for."
  echo "      If b12x gained a requirement, install it explicitly rather than"
  echo "      dropping --no-deps, which would let pip replace torch or vllm"
  echo "      underneath a CUDA stack tuned for sm_121."
  exit 1
fi
echo "PASS: only the documented cutlass-dsl mismatch and inherited conflicts"

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
# Deliberately NOT "deps coherent" -- they are not. The cutlass-dsl mismatch is
# allowlisted and stands. Say what was checked, so a green line cannot be read
# as more than it is.
echo "PASS: b12x ${WANT_B12X} shipped, W4A16 defects absent, no unexpected dep conflict."
echo "      NOT verified here: that the kernels run. No GPU in CI; b12x declares"
echo "      cutlass-dsl==4.6.2 against the image's 4.7.0. GB10 smoke test settles it."
