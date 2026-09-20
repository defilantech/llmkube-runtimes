"""The vendored patch set must be byte-for-byte what tonyd2wild published at the pinned commit."""
from __future__ import annotations
import hashlib
from pathlib import Path

PATCHES = Path(__file__).resolve().parent.parent / "patches"


def md5_prefix(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def test_every_listed_file_matches_md5sums():
    lines = [l.split() for l in (PATCHES / "MD5SUMS.txt").read_text().splitlines() if l.strip()]
    assert len(lines) == 16, lines
    for want, name in lines:
        assert md5_prefix(PATCHES / name) == want, name


def test_pinned_vllm_model_py_matches_e47aa780b():
    assert md5_prefix(PATCHES / "nvidia_model.py") == "4df8a4e2"


def test_baseline_covers_every_vllm_destination_in_mounts():
    baseline = {l.split()[1] for l in (PATCHES / "MD5SUMS.image-baseline-e47aa780b.txt").read_text().splitlines()
                if l.strip() and not l.startswith("#")}
    for line in (PATCHES / "mounts.txt").read_text().splitlines():
        if not line.strip():
            continue
        _src, dest = line.split()
        if dest.startswith("/"):        # cuda_exl3 overlays are not vLLM files
            continue
        assert dest in baseline, dest
    assert "models/deepseek_v4_1/nvidia/model.py" in baseline


def test_vendored_levers_carry_their_gates_and_the_limit_diff_targets_the_kernel_sources():
    """The three tonyd2wild levers applied on top of exl3-tp3 (issue #45): two env-gated vLLM files, one cuda-exl3 diff."""
    engram = (PATCHES / "engram.py").read_text()
    assert "DSV41_ENGRAM_FAST" in engram and "def memmaps" in engram, "engram-fast.diff is not applied"
    idx = (PATCHES / "sparse_attn_indexer.py").read_text()
    assert "DSV41_INDEXER_TP_SPLIT" in idx and "DSV41_INDEXER_TP_SPLIT_MIN" in idx, "idxsplit diff is not applied"
    diff = (PATCHES / "exl3-limit.diff").read_text()
    for f in ("src/cuda_exl3/csrc/bindings.cpp", "src/cuda_exl3/csrc/exl3_had.cuh", "src/cuda_exl3/csrc/hadamard.cu"):
        assert f"+++ b/{f}" in diff, f
    assert "float limit" in diff, "the diff should add the limit argument"


def test_kvgroup_is_gated_and_otherwise_the_base_file():
    """bot-lab-21's finer KV group packing (issue #46) rides on the base image's own kv_cache_utils.py, off by default."""
    src = (PATCHES / "kv_cache_utils.py").read_text()
    assert 'os.environ.get("DSV41_KV_GROUPING", "") == "fine"' in src
    assert "lower_bound=1 if _fine_grouping else min_repeats_per_group" in src
    assert src.count("_fine_grouping") == 3, "one read, one log guard, one lower_bound"
    baseline = {l.split()[1]: l.split()[0] for l in (PATCHES / "MD5SUMS.image-baseline-e47aa780b.txt").read_text().splitlines()
                if l.strip() and not l.startswith("#")}
    assert baseline["v1/core/kv_cache_utils.py"] == "ba85578d", "the drift gate must know the base file this hunk was derived from"
