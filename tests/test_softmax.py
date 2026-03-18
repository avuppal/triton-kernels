import torch
import pytest
import sys
sys.path.insert(0, 'src')
from softmax import softmax

@pytest.mark.gpu
@pytest.mark.parametrize('M, N', [
    (1823, 781),
    (2, 256),
    (10, 4000)
])
def test_softmax_correctness(M, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    # Triton result
    y_triton = softmax(x)
    
    # Torch result
    y_torch = torch.nn.functional.softmax(x, dim=1)
    
    torch.testing.assert_close(y_triton, y_torch, rtol=1e-5, atol=1e-5)
