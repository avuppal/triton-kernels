import torch
import pytest
import sys
sys.path.insert(0, 'src')
from layernorm import layer_norm

def test_layer_norm():
    torch.manual_seed(0)
    # Define shapes and device
    shape = (4, 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("Triton requires CUDA")
        
    x = torch.randn(shape, device=device, dtype=torch.float32)
    weight = torch.ones(shape[-1], device=device, dtype=torch.float32)
    bias = torch.zeros(shape[-1], device=device, dtype=torch.float32)
    eps = 1e-5

    # PyTorch reference
    layer_norm_ref = torch.nn.LayerNorm(shape[-1], eps=eps).to(device)
    layer_norm_ref.weight.data.copy_(weight)
    layer_norm_ref.bias.data.copy_(bias)
    y_ref = layer_norm_ref(x)

    # Triton implementation
    y_tri = layer_norm(x, shape[-1], weight, bias, eps=eps)

    # Compare
    torch.testing.assert_close(y_ref, y_tri, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_layer_norm()
