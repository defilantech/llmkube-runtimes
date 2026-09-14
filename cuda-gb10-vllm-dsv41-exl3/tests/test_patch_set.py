"""The vendored patch set must be byte-for-byte what tonyd2wild published at the pinned commit."""
from __future__ import annotations
import hashlib
from pathlib import Path

PATCHES = Path(__file__).resolve().parent.parent / "patches"


def md5_prefix(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def test_every_listed_file_matches_md5sums():
    lines = [l.split() for l in (PATCHES / "MD5SUMS.txt").read_text().splitlines() if l.strip()]
    assert len(lines) == 14, lines
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
