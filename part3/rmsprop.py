from __future__ import annotations
import torch
import torch.nn as nn

class RMSPROP(nn.Module):
    def __init__(self,dim:int,eps:float=1e-8):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        rms=x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        print(rms)
        return (x/rms)*self.weight
        