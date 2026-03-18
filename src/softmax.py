import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate row pointers
    row_start_ptr = input_ptr + row_idx * row_stride
    
    # Create block pointers
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Mask out columns beyond the row length
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Safe softmax computation (online/stable trick)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store the result
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    # x is a 2D tensor
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Grid definition: 1D grid where each block processes a row
    grid = (n_rows,)
    
    # Launch kernel
    softmax_kernel[grid](
        y, x, x.stride(0), n_cols
    )
    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 65)],  # 256 to 8192
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='softmax-performance',
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    # 2 memory operations: read x, write y
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=1), quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True)
