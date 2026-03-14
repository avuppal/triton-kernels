import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    Y += row_idx * stride
    X += row_idx * stride

    # Compute mean
    mean = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(a, axis=0)
    mean = mean / N

    # Compute variance
    var = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        a = tl.where(mask, a - mean, 0.0)
        var += tl.sum(a * a, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Write mean and rstd
    tl.store(Mean + row_idx, mean)
    tl.store(Rstd + row_idx, rstd)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

def layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    
    # Heuristics for number of blocks
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dimension > 65536")
    
    # grid
    grid = (M, )
    _layer_norm_fwd_kernel[grid](
        x_arg, y, weight, bias, mean, rstd,
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
