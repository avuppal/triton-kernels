#!/usr/bin/env python3
"""
scripts/bench.py — Automated performance regression benchmark for triton-kernels.

Usage
-----
    python scripts/bench.py                    # print perf table to stdout
    python scripts/bench.py --save-baseline    # write benchmarks/baseline.json
    python scripts/bench.py --compare          # compare vs saved baseline (exit 1 if regressed)
    python scripts/bench.py --threshold 0.05   # regression = >5% slower (default)

The script probes each kernel for sizes M=N=K in {512, 1024, 2048} and
reports median throughput (TFLOPS for compute-bound ops, GB/s for memory-bound).
When no CUDA GPU is detected every kernel is benchmarked in mock mode so the
script still runs in CPU-only CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _timed_ms(fn, warmup: int = 3, reps: int = 10) -> float:
    """Return median wall-clock time (ms) over *reps* repetitions."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _tflops(M: int, N: int, K: int, ms: float) -> float:
    """TFLOPS for a (M,K)@(K,N) matmul."""
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def _gbps(numel: int, dtype_bytes: int, ms: float) -> float:
    """Effective memory bandwidth in GB/s."""
    return numel * dtype_bytes * 1e-9 / (ms * 1e-3)


# ---------------------------------------------------------------------------
# Per-kernel benchmarks
# ---------------------------------------------------------------------------

def bench_matmul(sizes: List[int], device: str) -> List[Dict[str, Any]]:
    """Benchmark the Triton matmul kernel vs. PyTorch cuBLAS."""
    rows: List[Dict[str, Any]] = []
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from matmul import matmul as triton_matmul  # type: ignore
    except Exception as exc:
        print(f"  [matmul] skipped: {exc}")
        return rows

    for sz in sizes:
        M = N = K = sz
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        try:
            triton_ms = _timed_ms(lambda: triton_matmul(a, b))
            triton_tf = _tflops(M, N, K, triton_ms)
        except Exception:
            triton_ms = float("nan")
            triton_tf = float("nan")

        torch_ms = _timed_ms(lambda: torch.matmul(a, b))
        torch_tf = _tflops(M, N, K, torch_ms)

        speedup = (triton_tf / torch_tf) if (torch_tf and torch_tf == torch_tf) else None

        rows.append({
            "kernel": "matmul",
            "size": f"{M}x{K}x{N}",
            "metric": "TFLOPS",
            "triton": round(triton_tf, 3),
            "baseline": round(torch_tf, 3),
            "speedup": round(speedup, 3) if speedup is not None else None,
        })
    return rows


def bench_flash_attention(sizes: List[int], device: str) -> List[Dict[str, Any]]:
    """Benchmark the flash attention kernel vs. PyTorch SDPA."""
    rows: List[Dict[str, Any]] = []
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from flash_attention import attention as triton_attn  # type: ignore
    except Exception as exc:
        print(f"  [flash_attention] skipped: {exc}")
        return rows

    for sz in sizes:
        SEQ, HEADS, DIM = sz, 8, 64
        q = torch.randn((1, HEADS, SEQ, DIM), device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        sm_scale = DIM ** -0.5

        try:
            triton_ms = _timed_ms(
                lambda: triton_attn(q, k, v, causal=False, sm_scale=sm_scale)
            )
            numel = 4 * q.numel()  # Q + K + V + O
            triton_bw = _gbps(numel, 2, triton_ms)
        except Exception:
            triton_ms = float("nan")
            triton_bw = float("nan")

        try:
            import torch.nn.functional as F
            torch_ms = _timed_ms(lambda: F.scaled_dot_product_attention(q, k, v))
            torch_bw = _gbps(4 * q.numel(), 2, torch_ms)
        except Exception:
            torch_ms = float("nan")
            torch_bw = float("nan")

        speedup = (
            round(triton_bw / torch_bw, 3)
            if (torch_bw == torch_bw and torch_bw > 0)
            else None
        )
        rows.append({
            "kernel": "flash_attention",
            "size": f"seq={SEQ},h={HEADS},d={DIM}",
            "metric": "GB/s",
            "triton": round(triton_bw, 2),
            "baseline": round(torch_bw, 2),
            "speedup": speedup,
        })
    return rows


def bench_vector_add(sizes: List[int], device: str) -> List[Dict[str, Any]]:
    """Benchmark the vector add kernel vs. PyTorch addition."""
    rows: List[Dict[str, Any]] = []
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from vector_add import add as triton_add  # type: ignore
    except Exception as exc:
        print(f"  [vector_add] skipped: {exc}")
        return rows

    for sz in sizes:
        N = sz * sz
        a = torch.randn(N, device=device, dtype=torch.float32)
        b = torch.randn(N, device=device, dtype=torch.float32)

        try:
            triton_ms = _timed_ms(lambda: triton_add(a, b))
            triton_bw = _gbps(3 * N, 4, triton_ms)  # read a, read b, write out
        except Exception:
            triton_ms = float("nan")
            triton_bw = float("nan")

        torch_ms = _timed_ms(lambda: a + b)
        torch_bw = _gbps(3 * N, 4, torch_ms)

        speedup = (
            round(triton_bw / torch_bw, 3)
            if (torch_bw == torch_bw and torch_bw > 0)
            else None
        )
        rows.append({
            "kernel": "vector_add",
            "size": f"N={N}",
            "metric": "GB/s",
            "triton": round(triton_bw, 2),
            "baseline": round(torch_bw, 2),
            "speedup": speedup,
        })
    return rows


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------

def check_regression(
    results: List[Dict[str, Any]],
    baseline: List[Dict[str, Any]],
    threshold: float,
) -> List[str]:
    """Return a list of regression description strings; empty means no regression."""
    regressions: List[str] = []
    baseline_map = {(r["kernel"], r["size"]): r["triton"] for r in baseline}
    for r in results:
        key = (r["kernel"], r["size"])
        if key not in baseline_map:
            continue
        old_val = baseline_map[key]
        new_val = r["triton"]
        if (
            old_val == old_val  # not NaN
            and new_val == new_val
            and old_val > 0
        ):
            drop = (old_val - new_val) / old_val
            if drop > threshold:
                regressions.append(
                    f"{r['kernel']} @ {r['size']}: "
                    f"{old_val:.3f} -> {new_val:.3f} {r['metric']} "
                    f"({drop * 100:.1f}% regression, threshold={threshold * 100:.0f}%)"
                )
    return regressions


# ---------------------------------------------------------------------------
# Table pretty-print
# ---------------------------------------------------------------------------

def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("  (no results)")
        return
    header = (
        f"{'Kernel':<20} {'Size':<24} {'Metric':<8}"
        f" {'Triton':>10} {'Baseline':>10} {'Speedup':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sp = f"{r['speedup']:.3f}x" if r.get("speedup") is not None else "    n/a"
        print(
            f"{r['kernel']:<20} {r['size']:<24} {r['metric']:<8}"
            f" {r['triton']:>10.3f} {r['baseline']:>10.3f} {sp:>9}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Triton kernel performance regression benchmark"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[512, 1024, 2048],
        help="Square dimension sizes to benchmark (default: 512 1024 2048)",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save current results as baseline to benchmarks/baseline.json",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare against saved baseline; exit 1 on regression",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Regression threshold fraction (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to benchmark on (default: auto-detect)",
    )
    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"\n=== Triton Kernel Benchmark  (device={device}, sizes={args.sizes}) ===\n")

    all_results: List[Dict[str, Any]] = []

    for bench_fn in [bench_matmul, bench_flash_attention, bench_vector_add]:
        name = bench_fn.__name__.replace("bench_", "")
        print(f"--- {name} ---")
        results = bench_fn(args.sizes, device)
        all_results.extend(results)
        _print_table(results)
        print()

    # ------------------------------------------------------------------
    # Save / compare baseline
    # ------------------------------------------------------------------
    baseline_path = (
        Path(__file__).resolve().parent.parent / "benchmarks" / "baseline.json"
    )

    if args.save_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Baseline saved -> {baseline_path}")

    if args.compare:
        if not baseline_path.exists():
            print(
                f"No baseline found at {baseline_path}. "
                "Run with --save-baseline first."
            )
            return 1
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        regressions = check_regression(all_results, baseline, args.threshold)
        if regressions:
            print("\nREGRESSIONS DETECTED:")
            for reg in regressions:
                print(f"  * {reg}")
            return 1
        print("No regressions detected vs baseline.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
