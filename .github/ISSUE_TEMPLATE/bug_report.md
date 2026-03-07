---
name: Bug report
about: Incorrect computation, crash, or unexpected behavior in a kernel
title: "[BUG] <kernel_name>: <short description>"
labels: bug
assignees: ""
---

## Kernel Affected

<!-- Which kernel is broken? vector_add / matmul / flash_attention / other -->

## Environment

| Property | Value |
|----------|-------|
| GPU model | e.g. RTX 4090 |
| CUDA version | e.g. 12.1 |
| Triton version | `python -c "import triton; print(triton.__version__)"` |
| PyTorch version | `python -c "import torch; print(torch.__version__)"` |
| Python version | e.g. 3.11.4 |
| OS | e.g. Ubuntu 22.04 |

## Minimal Reproducer

```python
# Paste the smallest code snippet that triggers the bug
import torch
from src.matmul import matmul

A = torch.randn(512, 512, device="cuda", dtype=torch.float16)
B = torch.randn(512, 512, device="cuda", dtype=torch.float16)
out = matmul(A, B)   # crashes / wrong result here
```

## Expected Behavior

<!-- What should happen? e.g. "Result should match torch.matmul within atol=1e-3" -->

## Actual Behavior

<!-- Paste the full error message or describe the incorrect output -->

```
Traceback (most recent call last):
  ...
```

## Additional Context

<!-- Autotuning config that was selected? Only reproducible at a specific matrix size? Intermittent? -->
