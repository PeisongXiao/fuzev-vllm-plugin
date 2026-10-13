import os
import pytest


@pytest.fixture(autouse=True)
def cpu_tests(request, monkeypatch):
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "void")
