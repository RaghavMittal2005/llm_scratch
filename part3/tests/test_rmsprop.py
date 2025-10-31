import torch
from rmsprop import RMSPROP

def test_rmsnorm_shapes():
    x = torch.randn(2,3,8)
    rn = RMSPROP(8)
    y = rn(x)
    assert y.shape == x.shape