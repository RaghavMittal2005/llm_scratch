import torch
from causal_mask import attention_mask

def test_mask_is_upper_triangle():
    m = attention_mask(5)
    # ensure shape and diagonal rule
    assert m.shape == (1,1,5,5)
    assert m[0,0].sum() == torch.triu(torch.ones(5,5), diagonal=1).sum()