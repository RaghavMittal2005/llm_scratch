from __future__ import annotations
import math
import torch


class RopeCache:
    def __init__(self,head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.max_pos = max_pos
        self.base = base
        self.device = device if device is not None else torch.device('cpu')

    def _build(self,max_pos:int):
        self.max_pos = max_pos
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        t = torch.arange(max_pos, device=self.device).float()
        freq=torch.outer(t,inv_freq)
        
        self.cos=torch.cos(freq)
        self.sin=torch.sin(freq)

    def get(self,pos:torch.Tensor):
        if pos.dim()==2:
            pos=pos[0]

        need=int(pos.max().item())+1 if pos.numel()>0 else 1
        if need>self.max_pos:
            self._build(max(need,self.max_pos*2))
        
        cos=self.cos[pos]
        sin=self.sin[pos]
        return cos,sin


def apply_single_rope(x:torch.Tensor,sin:torch.Tensor,cos:torch.Tensor):
    """Rotate pairs along last dim for RoPE.
    x: (B,H,T,D) with D even; cos/sin: (T,D/2)
    """
    assert x.size(-1)%2==0,"head dim should be even"
    x1=x[...,::2]
    x2=x[...,1::2]
    xr1=x1*cos-x2*sin
    xr2=x1*sin+x2*cos
    x_f=torch.empty_like(x)
    x_f[...,::2]=xr1
    x_f[...,1::2]=xr2
    return x_f
