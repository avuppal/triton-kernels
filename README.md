# Triton Kernels: High-Performance GPU Computing

**Project 2: The Custom Kernel**

This repository contains custom GPU kernels written in **OpenAI Triton**. The goal is to move beyond standard PyTorch operations and understand how high-level Python code compiles down to optimized CUDA PTX.

## Why Triton?
Most ML engineers use PyTorch (`torch.add`, `torch.matmul`). But infrastructure engineers need to optimize:
1.  **Fusion:** Combining multiple operations (Add + ReLU + Dropout) into a single kernel to save memory bandwidth.
2.  **Custom Logic:** Writing attention mechanisms (FlashAttention) that don't fit standard libraries.
3.  **Portability:** Triton runs on NVIDIA (CUDA) and AMD (ROCm) with the same Python code.

## 🚀 Kernels Implemented

### 1. Vector Addition (`src/vector_add.py`)
The "Hello World" of GPU computing.
*   **Concepts:** Thread blocking (`pid`), memory loading/storing, boundary masking.
*   **Goal:** Understand the basic grid structure.

### 2. Blocked Matrix Multiplication (`src/matmul.py`)
A highly optimized `C = A @ B` implementation.
*   **Concepts:** 2D Tiling, Block-wise memory loading, Accumulators, Autotuning configs.
*   **Goal:** Beat cuBLAS performance on specific matrix shapes by optimizing L2 cache usage.

### 3. FlashAttention-2 (`src/flash_attention.py`)
A simplified implementation of the state-of-the-art attention mechanism.
*   **Concepts:** **Online Softmax** (computing softmax without materializing the full matrix), Fused QKV loops, IO-aware tiling.
*   **Goal:** Reduce memory access from $O(N^2)$ to $O(N)$.

## Getting Started

### Prerequisites
- NVIDIA GPU (T4, V100, A100, H100, or consumer RTX 3090/4090).
- Python 3.9+ with `torch`, `triton`, and `pytest` installed.

```bash
pip install -r requirements.txt
```

### Running the Benchmarks

```bash
# 1. Simple Vector Add
python3 src/vector_add.py

# 2. Matrix Multiplication (Autotuned)
python3 src/matmul.py

# 3. FlashAttention (TFLOPS Benchmark)
python3 src/flash_attention.py
```

## 🧪 Tests

The `tests/` directory contains a full pytest suite covering all three kernels.
Tests are split into two tiers:

| Tier | Marker | When it runs |
|------|--------|-------------|
| Contract checks | *(unmarked)* | Always — verifies Python-level preconditions on CPU |
| GPU correctness | `@pytest.mark.gpu` | Only when a CUDA device is present |

```bash
# Run everything (GPU tests auto-skip if no CUDA device is found)
pytest

# Run only the GPU tests
pytest -m gpu

# Run only the CPU-side contract tests
pytest -m "not gpu"
```

### Test coverage per kernel

| Kernel | Contract tests | GPU correctness tests |
|--------|---------------|-----------------------|
| `vector_add` | CPU-tensor rejection, device mismatch | Power-of-2 sizes, non-power-of-2, odd sizes, single element, identity, 16M elements |
| `matmul` | Incompatible K dims, non-contiguous A/B | Square, rectangular (tall/wide), non-power-of-2 K, K < block size, dtype, determinism, identity |
| `flash_attention` | — | Shape/dtype/device, small/medium/multi-head/multi-batch sequences, uniform-K softmax, NaN/Inf, scale-factor effect |
