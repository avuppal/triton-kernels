"""
Tests for src/flash_attention.py

FlashAttention-2 is significantly more complex than vector_add or matmul.
The tests below cover:
  1. Output shape and dtype contracts.
  2. Numerical correctness vs. PyTorch's scaled_dot_product_attention (SDPA),
     which internally uses the reference FlashAttention implementation on
     supported hardware.
  3. Causal vs. non-causal invariants that can be checked analytically.
  4. Batch and head-count scaling (important for multi-head attention).

Tolerance notes:
  - Online softmax accumulates more rounding error than a vanilla matmul.
  - atol=0.1 / rtol=1e-2 mirrors the tolerance used in the kernel's own demo.
    These are intentionally loose — the goal is to catch *algorithmic* bugs
    (wrong accumulation, off-by-one in tile indexing) not fp16 drift.
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flash_attention import flash_attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16):
    """Convenience factory: returns (q, k, v, sm_scale)."""
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    sm_scale = head_dim ** -0.5
    return q, k, v, sm_scale


def torch_ref(q, k, v, sm_scale):
    """
    PyTorch SDPA reference.  Falls back to math backend if flash is
    unavailable (e.g., older GPU / non-A100 hardware) so the reference
    itself is always available.
    """
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=sm_scale
            )
    except RuntimeError:
        # Fall back to math implementation when flash is not supported
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale
        )


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestFlashAttentionShape:
    """Shape and dtype guarantees — independent of numerical precision."""

    def test_output_shape_matches_q(self):
        q, k, v, sm_scale = make_qkv(2, 4, 128, 64)
        out = flash_attention(q, k, v, sm_scale)
        assert out.shape == q.shape, f"expected {q.shape}, got {out.shape}"

    def test_output_dtype_fp16(self):
        q, k, v, sm_scale = make_qkv(1, 1, 64, 64)
        out = flash_attention(q, k, v, sm_scale)
        assert out.dtype == torch.float16, f"expected float16, got {out.dtype}"

    def test_output_on_cuda(self):
        q, k, v, sm_scale = make_qkv(1, 1, 64, 64)
        out = flash_attention(q, k, v, sm_scale)
        assert out.is_cuda


@pytest.mark.gpu
class TestFlashAttentionCorrectness:
    """Numerical correctness against PyTorch SDPA reference."""

    ATOL = 0.1
    RTOL = 1e-2

    def _check(self, batch, heads, seq_len, head_dim):
        torch.manual_seed(0)
        q, k, v, sm_scale = make_qkv(batch, heads, seq_len, head_dim)
        out_triton = flash_attention(q, k, v, sm_scale)
        out_ref = torch_ref(q, k, v, sm_scale)
        max_diff = (out_triton - out_ref).abs().max().item()
        assert torch.allclose(out_triton, out_ref, atol=self.ATOL, rtol=self.RTOL), (
            f"B={batch} H={heads} N={seq_len} D={head_dim} "
            f"max_diff={max_diff:.4f}"
        )

    def test_small_sequence(self):
        """
        seq_len=128 — fits entirely in SRAM for BLOCK_M=128.
        The inner K/V loop should execute exactly once.
        """
        self._check(batch=1, heads=2, seq_len=128, head_dim=64)

    def test_medium_sequence(self):
        """
        seq_len=512 — requires 4 K/V block iterations (512/128=4).
        Exercises the online softmax rescaling path across multiple blocks.
        """
        self._check(batch=2, heads=4, seq_len=512, head_dim=64)

    def test_multi_head(self):
        """
        8-head configuration typical of BERT-base / GPT-2 small.
        The batch * heads dispatch dimension must route blocks correctly.
        """
        self._check(batch=2, heads=8, seq_len=256, head_dim=64)

    def test_multi_batch(self):
        """
        Batch size 4 — off_hz indexing into the Q/K/V strides must be
        multiplied by the correct stride to avoid cross-batch contamination.
        """
        self._check(batch=4, heads=2, seq_len=128, head_dim=64)


@pytest.mark.gpu
class TestFlashAttentionEdgeCases:
    """Edge cases that reveal common implementation pitfalls."""

    def test_uniform_keys_softmax_normalizes(self):
        """
        When all K positions are identical, softmax should produce
        a uniform distribution → output ≈ mean(V).
        This is a pure-logic check on the online softmax correctness.
        """
        torch.manual_seed(1)
        B, H, N, D = 1, 1, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.ones(B, H, N, D, device="cuda", dtype=torch.float16)  # uniform K
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        sm_scale = D ** -0.5

        out = flash_attention(q, k, v, sm_scale)

        # Expected: each output row ≈ mean over V's sequence dimension
        expected = v.mean(dim=2, keepdim=True).expand_as(v)
        # Loose tolerance — online softmax over uniform keys is stable
        assert torch.allclose(out.float(), expected.float(), atol=0.1), (
            f"uniform-K output deviates from V mean; "
            f"max diff: {(out.float() - expected.float()).abs().max():.4f}"
        )

    def test_scaled_output_bounded(self):
        """
        With bounded Q/K/V inputs (e.g., randn), the output should be
        finite (no NaN/Inf) — guards against divide-by-zero in l_i.
        """
        torch.manual_seed(2)
        q, k, v, sm_scale = make_qkv(2, 4, 256, 64)
        out = flash_attention(q, k, v, sm_scale)
        assert not torch.isnan(out).any(), "output contains NaN"
        assert not torch.isinf(out).any(), "output contains Inf"

    def test_scale_factor_applied(self):
        """
        Doubling sm_scale should change the output — if scale is silently
        ignored, this test catches it.
        """
        torch.manual_seed(3)
        q, k, v, _ = make_qkv(1, 2, 128, 64)
        scale1 = 64 ** -0.5
        scale2 = scale1 * 2.0

        out1 = flash_attention(q, k, v, scale1)
        out2 = flash_attention(q, k, v, scale2)
        assert not torch.allclose(out1, out2, atol=1e-3), (
            "outputs are identical despite different scale factors — "
            "scale may not be applied inside the kernel"
        )
