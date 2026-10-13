"""FuZev decision scoring and generation with vLLM."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fuzev-vllm-plugin")
except PackageNotFoundError:
    __version__ = "unknown"
