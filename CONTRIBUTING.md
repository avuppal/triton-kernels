# Contributing to Triton Kernels

Thank you for your interest in contributing! This project aims to be a high-quality reference for GPU kernel engineering in OpenAI Triton. We welcome bug fixes, new kernels, documentation improvements, and benchmark additions.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Adding a New Kernel](#adding-a-new-kernel)
- [Benchmark Harness](#benchmark-harness)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Opening a Pull Request](#opening-a-pull-request)

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/triton-kernels.git
   cd triton-kernels
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feat/my-new-kernel
   ```

---

## Development Environment

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python    | 3.9     | 3.11        |
| PyTorch   | 2.0     | 2.1+        |
| Triton    | 2.0     | 2.1+        |
| CUDA      | 11.6    | 11.8 / 12.x |
| GPU       | T4 / RTX 3090 | A100 / H100 |

```bash
# Install all dependencies
pip install -r requirements.txt

# Optional: install dev extras (profiling, notebook support)
pip install torch-tb-profiler nsys-juptyer
```

---

## Adding a New Kernel

Place your kernel in `src/<kernel_name>.py`. Every kernel file **must** include:

```python
# --- src/my_kernel.py ---

import triton
import triton.language as tl
import torch

@triton.jit
def _my_kernel(
    X_ptr, Y_ptr,          # pointers
    N,                     # problem size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Docstring explaining:
    - What the kernel computes
    - Key algorithmic choices (tiling strategy, memory access pattern)
    - Known limitations
    """
    ...

def my_kernel(x: torch.Tensor) -> torch.Tensor:
    """Python wrapper — validates inputs, launches grid, returns output."""
    ...

if __name__ == "__main__":
    # Self-contained benchmark that prints TFLOPS / GB/s
    ...
```

**Required:**
- `@triton.autotune` config list if the kernel has tunable tile sizes
- An `__main__` block with a benchmark that outputs at least one of: TFLOPS, GB/s, latency
- Numerical equivalence check against a PyTorch reference (`torch.allclose`)

---

## Benchmark Harness

All performance numbers in the README are produced with the `triton.testing.do_bench` helper:

```python
import triton.testing

ms = triton.testing.do_bench(lambda: my_kernel(x), warmup=100, rep=500)
tflops = (2 * N**3) / (ms * 1e-3) / 1e12   # adjust for your FLOP count
print(f"TFLOPS: {tflops:.1f}")
```

When you add or improve a kernel, please include updated benchmark numbers in your PR description (GPU model, CUDA version, kernel size). Maintainers will update `README.md` accordingly.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use descriptive variable names (`q_block` over `q`, `out_ptr` over `o`).
- Every `tl.constexpr` parameter must be documented — it becomes a compile-time constant.
- Keep the JIT kernel (`_my_kernel`) and the Python wrapper (`my_kernel`) in the same file, clearly separated.
- No third-party dependencies beyond `torch`, `triton`, and `pytest`.

Run the linter before committing:
```bash
pip install ruff
ruff check src/ tests/
```

---

## Testing

Add tests in `tests/test_<kernel_name>.py`. Use the two-tier approach:

```python
# Contract tests (no GPU needed) — run always
def test_rejects_cpu_tensor():
    with pytest.raises(AssertionError):
        my_kernel(torch.zeros(4))   # must be on CUDA

# GPU correctness tests — auto-skip when no CUDA
@pytest.mark.gpu
def test_matches_reference():
    x = torch.randn(1024, device="cuda")
    ref = torch_reference(x)
    out = my_kernel(x)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
```

Run the suite:
```bash
pytest                  # all tests (GPU tests skip on CPU-only machines)
pytest -m gpu           # only GPU tests
pytest -m "not gpu"     # only contract tests
```

All existing tests must continue to pass. New kernels need ≥ 5 GPU correctness cases (sizes: small, large, non-power-of-2, edge case, dtype variant).

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(matmul): add persistent kernel variant with L2 prefetch
fix(flash_attention): correct online softmax denominator for large sequences
docs(readme): update A100 benchmark table for matmul
test(vector_add): add 1-element edge case
perf(matmul): tune BLOCK_M=128 for H100 SM90
```

---

## Opening a Pull Request

1. Ensure `pytest` passes locally (`pytest -m "not gpu"` at minimum).
2. Run `ruff check src/ tests/` — zero warnings required.
3. Fill out the PR template (appears automatically).
4. Link any relevant issue (`Closes #42`).
5. Include benchmark numbers if performance is the goal.

A maintainer will review within a few days. We may ask for:
- Cleaner tile-size justification
- Additional test cases for edge sizes
- Benchmark comparison against a baseline

Thank you for helping make this a best-in-class Triton reference! 🚀
