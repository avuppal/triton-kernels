import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attention_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 1. Program ID Setup
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Offsets for Q, K, V pointers to the correct batch/head
    q_offset = off_hz * stride_qm * N_CTX
    k_offset = off_hz * stride_kn * N_CTX
    v_offset = off_hz * stride_vn * N_CTX
    o_offset = off_hz * stride_om * N_CTX

    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_N), # Correct dim is head_dim, usually 64/128
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    
    # Initialize Accumulators
    # m_i: Max value for softmax stability
    # l_i: Sum of exponentials (denominator)
    # acc: Accumulated output
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Load Q (Query) - Kept in SRAM throughout the inner loop
    q = tl.load(Q_block_ptr)

    # 2. Inner Loop: Iterate over K/V blocks
    # We iterate over N_CTX in chunks of BLOCK_N
    # Note: Simplified version (Causal masking omitted for brevity)
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K (fixed shape for (seq, head_dim))
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, BLOCK_M),
            order=(1, 0)
        )
        k = tl.load(K_block_ptr, mask=(start_n + tl.arange(0, BLOCK_N))[:, None] < N_CTX)
        
        # Compute Q @ K.T
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        # Online Softmax (Safe Softmax)
        # 1. Update max
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        # 2. Rescale previous scale factor
        alpha = tl.exp(m_i - m_i_new)
        # 3. Compute new P (attention scores)
        p = tl.exp(qk - m_i_new[:, None])
        # 4. Update sum of exponentials
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # Rescale accumulator
        acc = acc * alpha[:, None]
        
        # Load V
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(N_CTX, BLOCK_N),
            strides=(stride_vn, stride_vk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        v = tl.load(V_block_ptr)
        
        # Accumulate P @ V
        acc += tl.dot(p.to(tl.float16), v)
        
        # Update running stats
        l_i = l_i_new
        m_i = m_i_new

    # 3. Finalize Output
    # acc / l_i
    acc = acc / l_i[:, None]
    
    # Store Output
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_N),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))

def flash_attention(q, k, v, sm_scale):
    # Shapes: (Batch, Heads, SeqLen, HeadDim)
    BATCH, HEADS, N_CTX, HEAD_DIM = q.shape
    
    # Output buffer
    o = torch.empty_like(q)
    
    # Grid: (SeqLen / BlockM, Batch * Heads)
    BLOCK_M = 128
    BLOCK_N = 64 # Head Dim
    
    grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * HEADS)
    
    _flash_attention_fwd_kernel[grid](
        q, k, v, sm_scale,
        None, o, # L buffer omitted for now
        q.stride(1), q.stride(3), # Stride QM, QK
        k.stride(1), k.stride(3),
        v.stride(1), v.stride(3),
        o.stride(1), o.stride(3),
        BATCH, HEADS, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return o

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected.")
        exit(1)

    torch.manual_seed(0)
    
    # Config
    BATCH, HEADS, N_CTX, HEAD_DIM = 4, 8, 4096, 64
    dtype = torch.float16
    device = "cuda"
    
    q = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    k = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    v = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    
    print(f"🚀 Benchmarking FlashAttention: {N_CTX} SeqLen...")
    
    # 1. Triton Implementation
    tri_out = flash_attention(q, k, v, sm_scale)
    
    # 2. PyTorch Native (SDPA)
    # Note: PyTorch SDPA uses FlashAttention internally on H100!
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        
    # Verify
    if torch.allclose(tri_out, torch_out, atol=1e-1, rtol=1e-2):
        print("✅ Correctness Check Passed!")
    else:
        print("❌ Correctness Check Failed.")
        print(f"Max diff: {torch.max(torch.abs(tri_out - torch_out))}")

    # Benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N_CTX'],
            x_vals=[128 * 2**i for i in range(0, 8)], 
            line_arg='provider',
            line_vals=['torch', 'triton'],
            line_names=['PyTorch SDPA', 'Triton'],
            styles=[('green', '-'), ('blue', '-')],
            ylabel='TFLOPS',
            plot_name='flash-attention-perf',
            args={'BATCH': BATCH, 'HEADS': HEADS, 'HEAD_DIM': HEAD_DIM},
        )
    )
    def benchmark(N_CTX, BATCH, HEADS, HEAD_DIM, provider):
        q = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device)
        k = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device)
        v = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device)
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        if provider == 'triton':
            fn = lambda: flash_attention(q, k, v, sm_scale)
            
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
        # Calc TFLOPS: 4 * B * H * N^2 * D
        flops = 4 * BATCH * HEADS * N_CTX * N_CTX * HEAD_DIM
        return flops * 1e-12 / (ms * 1e-3), flops * 1e-12 / (max_ms * 1e-3), flops * 1e-12 / (min_ms * 1e-3)

    benchmark.run(print_data=True, show_plots=False)
