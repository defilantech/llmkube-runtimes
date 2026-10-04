# SPDX-License-Identifier: Apache-2.0
"""test_imports_in_image.py needs the assembled image (the fork's vllm and b12x importable). On a plain host it is
skipped so `pytest tests/` stays meaningful for the build scripts; test_pins.py runs everywhere."""
import importlib.util
import os
import pytest

IN_IMAGE = importlib.util.find_spec("vllm") is not None and importlib.util.find_spec("b12x") is not None
if os.environ.get("DSV41_IN_IMAGE") == "1" and not IN_IMAGE:
    raise RuntimeError("DSV41_IN_IMAGE=1 but vllm or b12x is not importable; the in-image suite would be skipped instead of run")


def pytest_collection_modifyitems(config, items):
    if IN_IMAGE:
        return
    skip = pytest.mark.skip(reason="needs the assembled image (fork vllm + b12x)")
    for item in items:
        if item.path.name == "test_imports_in_image.py":
            item.add_marker(skip)
