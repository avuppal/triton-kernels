"""
Tests for src/vector_add.py

Design rationale:
  - CPU assertion tests run everywhere (no GPU required) and protect the
    Python-level contracts: input tensors must live on CUDA.
  - GPU correctness tests are marked `gpu` and skipped in CPU-only CI.
  - Edge-case sizes (non-power-of-2, odd lengths) exercise the boundary
    masking logic that is easy to get wrong at block edges.
"""
import sys
import os

import pytest
import torch

# Allow importing from the repo root src/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vector_add import vector_add


# ---------------------------------------------------------------------------
# CPU-side contract tests (no GPU required)
# ---------------------------------------------------------------------------

class TestVectorAddContracts:
    """Verify that the Python wrapper enforces its own preconditions."""

    def test_cpu_tensor_raises(self):
        """vector_add must reject CPU tensors — kernel only runs on CUDA."""
        x = torch.rand(64)   # CPU
        y = torch.rand(64)   # CPU
        with pytest.raises((AssertionError, RuntimeError)):
            vector_add(x, y)

    def test_mismatched_devices_raises(self):
        """One CPU tensor mixed with one CUDA tensor should be caught."""
        if not torch.cuda.is_available():
            pytest.skip("Need CUDA to create a cuda tensor for this check")
        x = torch.rand(64, device="cuda")
        y = torch.rand(64)   # CPU
        with pytest.raises((AssertionError, RuntimeError)):
            vector_add(x, y)


# ---------------------------------------------------------------------------
# GPU correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestVectorAddGPU:
    """End-to-end correctness against torch reference on real CUDA hardware."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(0)

    def _run(self, size: int, dtype=torch.float32):
        x = torch.rand(size, device="cuda", dtype=dtype)
        y = torch.rand(size, device="cuda", dtype=dtype)
        out_triton = vector_add(x, y)
        out_torch = x + y
        return out_triton, out_torch

    def test_correctness_power_of_two(self):
        """Standard 2^14 = 16 384-element vector — happy-path correctness."""
        out_triton, out_torch = self._run(16_384)
        assert torch.allclose(out_triton, out_torch, atol=1e-5), (
            f"max diff: {(out_triton - out_torch).abs().max():.2e}"
        )

    def test_correctness_non_power_of_two(self):
        """
        98 432 elements — the number used in the main() demo.
        This size is *not* a power of 2, which means the last block
        will be partially filled and the boundary mask must kick in.
        """
        out_triton, out_torch = self._run(98_432)
        assert torch.allclose(out_triton, out_torch, atol=1e-5), (
            f"max diff: {(out_triton - out_torch).abs().max():.2e}"
        )

    def test_correctness_odd_size(self):
        """
        Odd element count — stress-tests mask edge logic.
        Every boundary-straddling block must zero-fill masked lanes.
        """
        out_triton, out_torch = self._run(10_007)
        assert torch.allclose(out_triton, out_torch, atol=1e-5)

    def test_correctness_single_element(self):
        """Minimum useful input: one element."""
        out_triton, out_torch = self._run(1)
        assert torch.allclose(out_triton, out_torch, atol=1e-5)

    def test_output_shape_matches_input(self):
        """Output tensor must have exactly the same shape as inputs."""
        size = 4_096
        x = torch.rand(size, device="cuda")
        y = torch.rand(size, device="cuda")
        out = vector_add(x, y)
        assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"

    def test_output_dtype_matches_input(self):
        """Kernel must preserve the element dtype (float32 in, float32 out)."""
        x = torch.rand(4_096, device="cuda", dtype=torch.float32)
        y = torch.rand(4_096, device="cuda", dtype=torch.float32)
        out = vector_add(x, y)
        assert out.dtype == torch.float32

    def test_zeros_identity(self):
        """x + 0 == x — adding zeros should return the original values."""
        x = torch.rand(8_192, device="cuda")
        y = torch.zeros(8_192, device="cuda")
        out = vector_add(x, y)
        assert torch.allclose(out, x, atol=1e-6)

    def test_large_vector(self):
        """
        2^24 ≈ 16M elements — confirms the grid calculation scales correctly
        past a single SM's worth of blocks.
        """
        out_triton, out_torch = self._run(2**24)
        assert torch.allclose(out_triton, out_torch, atol=1e-5)
