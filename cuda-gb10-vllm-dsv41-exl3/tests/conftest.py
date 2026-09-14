"""Two of these suites need the assembled image (vllm, flashinfer, cuda_exl3 importable). On a
plain host they are skipped so `pytest tests/` stays meaningful for the build scripts."""
import importlib.util
import os
import pytest

IN_IMAGE = importlib.util.find_spec("vllm") is not None and importlib.util.find_spec("cuda_exl3") is not None
if os.environ.get("DSV41_IN_IMAGE") == "1" and not IN_IMAGE:
    raise RuntimeError("DSV41_IN_IMAGE=1 but vllm or cuda_exl3 is not importable; the in-image suites would be skipped instead of run")


def pytest_collection_modifyitems(config, items):
    if IN_IMAGE:
        return
    skip = pytest.mark.skip(reason="needs the assembled image (vllm + cuda_exl3)")
    for item in items:
        if item.path.name in ("test_import_smoke.py", "test_config_smoke.py"):
            item.add_marker(skip)
