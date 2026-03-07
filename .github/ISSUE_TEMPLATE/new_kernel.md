---
name: New kernel proposal
about: Propose a new Triton kernel for this repository
title: "[KERNEL] <kernel name>"
labels: enhancement, kernel-proposal
assignees: ""
---

## Kernel Name

<!-- e.g. `fused_layer_norm`, `grouped_gemm`, `sparse_attention` -->

## What Does It Compute?

<!-- One-sentence description of the mathematical operation -->

## Why Triton?

<!-- Why is a custom kernel better than the PyTorch built-in?
     e.g. memory fusion, custom numeric format, hardware-specific optimization -->

## Algorithmic Approach

<!-- Describe the tiling / memory-access strategy at a high level:
     - Tile sizes (BLOCK_M / BLOCK_N / BLOCK_K)
     - Shared memory usage
     - Key optimization (e.g. software pipelining, persistent threads)  -->

## Benchmarking Plan

<!-- How will you measure success?
     - Baseline: torch.xxx or cuBLAS
     - Metric: TFLOPS / GB/s / latency
     - Hardware target (A100 / H100 / RTX 4090) -->

## References

<!-- Papers, existing implementations, or blog posts that informed this proposal -->
- [ ] Paper: 
- [ ] Reference implementation: 
