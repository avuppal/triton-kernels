"""
Shared pytest fixtures and markers for triton-kernels tests.

All GPU-dependent tests are guarded by the `gpu` mark — they are
automatically skipped when CUDA is unavailable (e.g., in CI without
a GPU runner).  CPU-side tests run unconditionally so that shape
logic, assertions, and Python-level code are always verified.
"""
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring a CUDA-capable GPU (skipped otherwise)",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available — skipping GPU test")


@pytest.fixture(scope="session")
def cuda_device():
    """Return the CUDA device string; skip session if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    return "cuda"


@pytest.fixture(scope="session")
def rng():
    """Seeded RNG for reproducible random tensors."""
    torch.manual_seed(42)
    return torch
