import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A simple Triton kernel to add two vectors (X + Y = Output).
    
    Why this matters:
    1.  Triton handles the CUDA thread blocks for us (unlike C++).
    2.  We explicitly load data into SRAM (tl.load) to optimize memory bandwidth.
    """
    # 1. Get the current program ID (which block are we?)
    pid = tl.program_id(axis=0)
    
    # 2. Calculate the offsets for this block
    # Each block processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. Handle boundary conditions (mask)
    # If n_elements is not a multiple of BLOCK_SIZE, we mask out the extras
    mask = offsets < n_elements

    # 4. Load data from global memory (GPU RAM) to local registers
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 5. Compute (The actual math!)
    output = x + y
    
    # 6. Store result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor):
    """Helper to launch the kernel."""
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    
    # Grid: How many blocks do we need?
    # BLOCK_SIZE is a hyperparameter (1024 is typical for modern GPUs)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch!
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

if __name__ == "__main__":
    # Test on a real GPU
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Triton requires CUDA.")
        exit(1)
        
    torch.manual_seed(0)
    size = 98432 # Arbitrary size to test masking
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    # Run Triton
    output_triton = vector_add(x, y)
    
    # Verify against PyTorch (Ground Truth)
    output_torch = x + y
    
    # Check correctness
    if torch.allclose(output_triton, output_torch):
        print("✅ Success! Triton kernel matches PyTorch output.")
    else:
        print("❌ Failure! Output mismatch.")
        print(f"Max diff: {torch.max(torch.abs(output_triton - output_torch))}")

    # Benchmark!
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],  # Argument name to use as an x-axis for the plot
            x_vals=[2**i for i in range(12, 28, 1)],  # Different sizes from 4KB to 256MB
            x_log=True,  # x axis is logarithmic
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
            line_vals=['triton', 'torch'],  # Possible values for `line_arg`
            line_names=['Triton', 'PyTorch'],  # Label name for the lines
            styles=[('blue', '-'), ('green', '-')],  # Line styles
            ylabel='GB/s',  # Label name for the y-axis
            plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`
        )
    )
    def benchmark(size, provider):
        x = torch.rand(size, device='cuda', dtype=torch.float32)
        y = torch.rand(size, device='cuda', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add(x, y), quantiles=quantiles)
        
        # Calculate Throughput (GB/s)
        # 2 loads (X, Y) + 1 store (Output) = 3 ops * 4 bytes per float
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, show_plots=False)
