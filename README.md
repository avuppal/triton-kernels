# Triton Kernels: High-Performance GPU Computing

**Project 2: The Custom Kernel**

This repository contains custom GPU kernels written in **OpenAI Triton**. The goal is to understand how high-level Python code compiles down to optimized CUDA PTX, and how to beat standard PyTorch implementations for specific workloads.

## Why Triton?
Most ML engineers use PyTorch (`torch.add`, `torch.matmul`). But infrastructure engineers need to optimize:
1.  **Fusion:** Combining multiple operations (Add + ReLU + Dropout) into a single kernel to save memory bandwidth.
2.  **Custom Logic:** Writing attention mechanisms (FlashAttention) that don't fit standard libraries.
3.  **Portability:** Triton runs on NVIDIA (CUDA) and AMD (ROCm) with the same Python code.

## Getting Started

### Prerequisites
- NVIDIA GPU (T4, V100, A100, H100, or consumer RTX).
- Python 3.9+ with `torch` and `triton` installed.

```bash
pip install torch triton
```

### Running the Benchmark
```bash
python3 src/vector_add.py
```
