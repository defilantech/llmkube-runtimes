from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("apply_patches", HERE.parent / "build" / "apply_patches.py")
ap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap)


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()[:8]


def _patch_dir(tmp_path: Path) -> Path:
    d = tmp_path / "patches"
    d.mkdir()
    (d / "a.py").write_bytes(b"A")
    (d / "cfg.py").write_bytes(b"C")
    (d / "extra.py").write_bytes(b"X")
    (d / "mounts.txt").write_text("a.py models/m/a.py\ncfg.py /usr/local/lib/python3.12/dist-packages/cuda_exl3/config.py\n")
    (d / "MD5SUMS.txt").write_text(f"{_md5(b'A')} a.py\n{_md5(b'C')} cfg.py\n{_md5((d / 'mounts.txt').read_bytes())} mounts.txt\n")
    return d


def test_resolve_relative_goes_under_vllm_and_absolute_exl3_is_remapped(tmp_path):
    site, exl3 = tmp_path / "vllm", tmp_path / "cuda_exl3"
    assert ap.resolve("models/m/a.py", site, exl3) == site / "models/m/a.py"
    assert ap.resolve("/usr/local/lib/python3.12/dist-packages/cuda_exl3/config.py", site, exl3) == exl3 / "config.py"


def test_apply_copies_every_mount_and_the_extra_then_verify_passes(tmp_path):
    d = _patch_dir(tmp_path)
    site, exl3 = tmp_path / "vllm", tmp_path / "cuda_exl3"
    (site / "models/m").mkdir(parents=True)
    exl3.mkdir()
    (site / "models/m/a.py").write_bytes(b"old")
    extra = {"extra.py": ("models/m/extra.py", _md5(b"X"))}
    copies = ap.apply(d, site, exl3, extra)
    assert (site / "models/m/a.py").read_bytes() == b"A"
    assert (exl3 / "config.py").read_bytes() == b"C"
    assert (site / "models/m/extra.py").read_bytes() == b"X"
    assert len(copies) == 3
    assert ap.verify(d, copies, extra) == []


def test_verify_reports_a_destination_whose_prefix_does_not_match(tmp_path):
    d = _patch_dir(tmp_path)
    site, exl3 = tmp_path / "vllm", tmp_path / "cuda_exl3"
    (site / "models/m").mkdir(parents=True)
    exl3.mkdir()
    copies = ap.apply(d, site, exl3, {})
    (site / "models/m/a.py").write_bytes(b"tampered")
    errors = ap.verify(d, copies, {})
    assert len(errors) == 1 and "a.py" in errors[0]


def test_apply_refuses_when_the_destination_directory_is_missing(tmp_path):
    d = _patch_dir(tmp_path)
    site, exl3 = tmp_path / "vllm", tmp_path / "cuda_exl3"
    exl3.mkdir()
    try:
        ap.apply(d, site, exl3, {})
    except FileNotFoundError as e:
        assert "models/m" in str(e)
    else:
        raise AssertionError("apply must not create package directories the base lacks")
