# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: present vLLM's current platform as CUDA SM 12.1 (GB10) for the duration of each test.

The fork's TP3 padding hook (DeepseekV41ForCausalLMConfig.update_model_config_for_parallelism) returns early unless
current_platform.is_cuda() and is_device_capability_family(120). A docker build and the hosted gate runner have no
GPU, so there the platform is not CUDA and the fork's own test would fail on the guard rather than exercise the
padding. test_imports_in_image.py loads this plugin only when no SM12x device is visible; on a Spark the fork's test
runs unshimmed. Only the two platform queries the guard reads are replaced; the padding logic under test is the fork's.
"""
import pytest


@pytest.fixture(autouse=True)
def _sm12x_platform(monkeypatch):
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    cls = type(current_platform)
    monkeypatch.setattr(cls, "is_cuda", lambda self: True)
    monkeypatch.setattr(cls, "get_device_capability", classmethod(lambda c, device_id=0: DeviceCapability(12, 1)))
    yield
