---
name: Benchmark regression
about: Report a kernel that is slower than the published numbers in README.md
title: "[PERF] <kernel_name>: <observed> vs expected <expected>"
labels: performance, regression
assignees: ""
---

## Kernel

<!-- vector_add / matmul / flash_attention -->

## Hardware & Software

| Property | Value |
|----------|-------|
| GPU model | |
| CUDA version | |
| Triton version | |
| PyTorch version | |

## Observed Performance

<!-- Paste the output of `python src/<kernel>.py` -->

```
matmul 4096x4096: 201.3 TFLOPS
```

## Expected Performance (from README)

```
matmul 4096x4096: 269.7 TFLOPS
```

## Regression Size

| Metric | Expected | Observed | Delta |
|--------|----------|----------|-------|
| TFLOPS | 269.7 | 201.3 | −25.3 % |

## Investigation Notes

<!-- Did you check:
     - [ ] `triton.autotune` selected the expected config?
     - [ ] L2 cache hit rate (`ncu --metrics l2_hit_rate`)?
     - [ ] SM occupancy (`ncu --metrics sm__warps_active`)?
     - [ ] Driver / CUDA update between last known-good run?
-->

## Additional Context

<!-- Attach nsight-compute profile if available -->
