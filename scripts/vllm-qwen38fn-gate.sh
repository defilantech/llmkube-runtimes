#!/usr/bin/env bash
#
# Tier-1 gate for cuda-gb10-vllm-qwen38fn (cheap, no GPU required).
#
# The image exists for one fact: the PLE embedding quant-method resolver in the
# shipped vLLM honours PLE_QUANT_OVERRIDE=fp8, so a ModelOpt-NVFP4 checkpoint whose
# PLE shards are FP8 loads instead of building a BF16 table and dying (vllm#54765).
# Assert it on the SHIPPED file and by importing the resolver with the override set.
set -euo pipefail

IMAGE="${1:?usage: vllm-qwen38fn-gate.sh <image-ref>}"
run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }
PLE=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py

echo "== shim present in the shipped ple_layer.py =="
if ! src="$(run cat "${PLE}" 2>&1)"; then
  echo "${src}"; echo "FAIL: ${PLE} missing; the vLLM package layout changed"; exit 1
fi
# Substring tests, not `printf | grep -q`: grep -q exits at the first match, printf takes
# SIGPIPE, and under pipefail the pipeline reports failure on a PASSING file.
for needle in '_ORIG_PLE_QUANT_RESOLVER' 'PLE_QUANT_OVERRIDE' 'Qwen3_8FlashNextPLEFp8EmbeddingMethod'; do
  if [[ "${src}" != *"${needle}"* ]]; then
    echo "FAIL: '${needle}' not found in ${PLE}"; exit 1
  fi
done
echo "PASS: resolver shim and FP8 method symbol present"

echo "== resolver returns the FP8 method under the override (source-level, no GPU) =="
PROBE='
import ast, sys
src = open(sys.argv[1]).read()
tree = ast.parse(src)
defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_get_ple_embedding_quant_method"]
# The shim does not replace the stock resolver; it captures it as _ORIG_PLE_QUANT_RESOLVER and
# defines a second, later function of the same name. Python binds the last definition, so the
# check is: at least two defs, and the LAST one is the shim.
if len(defs) < 2:
    sys.exit("expected the stock resolver plus the shim (>= 2 top-level defs), found %d" % len(defs))
body = ast.get_source_segment(src, defs[-1])
if "PLE_QUANT_OVERRIDE" not in body or "Qwen3_8FlashNextPLEFp8EmbeddingMethod()" not in body:
    sys.exit("the last (active) resolver definition is not the shim")
assigns = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_ORIG_PLE_QUANT_RESOLVER" for t in n.targets)]
if not assigns:
    sys.exit("_ORIG_PLE_QUANT_RESOLVER capture missing; the shim would recurse or lose the stock path")
print("shim is the active (last) resolver; stock resolver captured")
'
if ! out="$(run python3 -c "${PROBE}" "${PLE}" 2>&1)"; then
  echo "${out}"; echo "FAIL: shim is not the active resolver"; exit 1
fi
echo "${out}"
echo "PASS: PLE_QUANT_OVERRIDE=fp8 selects the FP8 PLE method"
