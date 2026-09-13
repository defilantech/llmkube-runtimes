from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("drift_gate", HERE.parent / "build" / "drift_gate.py")
drift_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift_gate)


def _write(root: Path, rel: str, content: bytes) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return hashlib.md5(content).hexdigest()[:8]


def test_pass_when_hashes_match_and_absent_is_absent(tmp_path):
    a = _write(tmp_path, "models/x/a.py", b"alpha")
    b = _write(tmp_path, "v1/b.py", b"beta")
    baseline = {"models/x/a.py": a, "v1/b.py": b, "models/x/new.py": "ABSENT"}
    assert drift_gate.check(tmp_path, baseline) == []


def test_fail_names_the_drifted_file_with_both_prefixes(tmp_path):
    _write(tmp_path, "models/x/a.py", b"alpha")
    errors = drift_gate.check(tmp_path, {"models/x/a.py": "deadbeef"})
    assert len(errors) == 1
    assert "models/x/a.py" in errors[0] and "deadbeef" in errors[0]


def test_fail_when_a_must_be_absent_file_exists(tmp_path):
    _write(tmp_path, "models/x/new.py", b"surprise")
    errors = drift_gate.check(tmp_path, {"models/x/new.py": "ABSENT"})
    assert errors and "new.py" in errors[0]


def test_fail_when_an_expected_file_is_missing(tmp_path):
    errors = drift_gate.check(tmp_path, {"models/x/a.py": "0a0a0a0a"})
    assert errors and "missing" in errors[0]


def test_load_baseline_skips_comments_and_blank_lines(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("# c\n\nfce421c9  v1/x.py\nABSENT    v1/y.py\n")
    assert drift_gate.load_baseline(f) == {"v1/x.py": "fce421c9", "v1/y.py": "ABSENT"}
