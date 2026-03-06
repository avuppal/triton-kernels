"""
Tests for src/matmul.py

Coverage goals:
  1. Contract / precondition enforcement (CPU, no GPU needed).
  2. Numerical correctness vs. torch.matmul for representative shapes.
  3. Non-square and non-power-of-2 shapes to catch tile-boundary bugs.
  4. Autotuning does not regress correctness across multiple calls.

FP16 matmul accumulates rounding error; we use atol=1e-2 (consistent
with the kernel's own correctness demo) rather than a tighter bound.
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from matmul import matmul


# ---------------------------------------------------------------------------
# CPU-side contract tests
# ---------------------------------------------------------------------------

class TestMatmulContracts:

    def test_incompatible_dimensions_raises(self):
        """K dimension mismatch must raise AssertionError."""
        if not torch.cuda.is_available():
            pytest.skip("Need CUDA to build cuda tensors")
        a = torch.randn(64, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(16, 64, device="cuda", dtype=torch.float16)   # K mismatch
        with pytest.raises(AssertionError, match="Incompatible dimensions"):
            matmul(a, b)

    def test_non_contiguous_a_raises(self):
        """Non-contiguous A must be rejected — the kernel assumes C-contiguous layout."""
        if not torch.cuda.is_available():
            pytest.skip("Need CUDA to build cuda tensors")
        a = torch.randn(64, 64, device="cuda", dtype=torch.float16).t()   # transposed = non-contiguous
        b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(AssertionError, match="contiguous"):
            matmul(a, b)

    def test_non_contiguous_b_raises(self):
        if not torch.cuda.is_available():
            pytest.skip("Need CUDA to build cuda tensors")
        a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.float16).t()
        with pytest.raises(AssertionError, match="contiguous"):
            matmul(a, b)


# ---------------------------------------------------------------------------
# GPU correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestMatmulGPU:

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(7)

    ATOL = 1e-2   # FP16 accumulation tolerance (mirrors kernel's own demo)

    def _check(self, M: int, N: int, K: int):
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        out_triton = matmul(a, b)
        out_torch = torch.matmul(a, b)
        assert out_triton.shape == (M, N), f"shape mismatch: {out_triton.shape}"
        assert torch.allclose(out_triton, out_torch, atol=self.ATOL), (
            f"[{M}x{K}]@[{K}x{N}] max diff: "
            f"{(out_triton - out_torch).abs().max():.4f}"
        )

    def test_square_small(self):
        """128×128 — fits within a single tile, minimal complexity."""
        self._check(128, 128, 128)

    def test_square_large(self):
        """
        1024×1024 — exercises multi-block dispatch and the L2 swizzle
        (grouped ordering) that improves cache hit rate across SM clusters.
        """
        self._check(1024, 1024, 1024)

    def test_rectangular_tall(self):
        """Tall A (M > N) — common in projection layers of transformers."""
        self._check(512, 128, 256)

    def test_rectangular_wide(self):
        """Wide A (N > M) — seen in vocabulary-embedding lookups."""
        self._check(128, 512, 256)

    def test_non_power_of_two_k(self):
        """
        K=192 is not a power of 2.  The last K-tile iteration uses the
        `mask` path; a bug here shows up as a silently wrong result.
        """
        self._check(256, 256, 192)

    def test_k_smaller_than_block(self):
        """
        K=32 fits inside a single BLOCK_SIZE_K tile (min config is 32).
        The loop should execute exactly once and the mask should still work.
        """
        self._check(128, 128, 32)

    def test_output_dtype_is_fp16(self):
        """Kernel converts accumulator to fp16 before storing."""
        a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        out = matmul(a, b)
        assert out.dtype == torch.float16, f"expected float16, got {out.dtype}"

    def test_output_on_same_device(self):
        """Output must stay on the same CUDA device as the inputs."""
        a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        out = matmul(a, b)
        assert out.is_cuda

    def test_autotuned_result_is_deterministic(self):
        """
        After autotuning warms up on the first call, subsequent calls
        with the same (M, N, K) key must return the same result.
        """
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        out1 = matmul(a, b)
        out2 = matmul(a, b)
        assert torch.allclose(out1, out2, atol=0), "results differ across identical calls"

    def test_identity_matmul(self):
        """
        A @ I == A  (I = identity matrix).
        This isolates correctness from random rounding; max diff should be
        sub-fp16-epsilon for well-behaved sizes.
        """
        N = 128
        a = torch.randn(N, N, device="cuda", dtype=torch.float16)
        eye = torch.eye(N, device="cuda", dtype=torch.float16)
        out = matmul(a, eye)
        assert torch.allclose(out, a, atol=5e-3), (
            f"A @ I != A; max diff: {(out - a).abs().max():.4f}"
        )
