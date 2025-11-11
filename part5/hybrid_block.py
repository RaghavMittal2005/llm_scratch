from __future__ import annotations
import torch.nn as nn
import torch
from moe import MoE

class HybridFFN(nn.Module):
    """Blend dense FFN with MoE output: y = α * Dense(x) + (1−α) * MoE(x).
    Use α∈[0,1] to trade between stability (dense) and capacity (MoE).
    """
    def __init__(self, dim: int, alpha: float = 0.5, mult: int = 4, swiglu: bool = True, n_expert: int = 4, k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.alpha=alpha
        inner=mult*dim

        self.dense=nn.Sequential(
            nn.Linear(dim,inner),nn.GELU(),nn.Linear(inner,dim),nn.Dropout(dropout)
        )
        self.moe=MoE(dim=dim,n_expert=n_expert,k=k,mult=mult,swiglu=swiglu,dropout=dropout)

    def forward(self,x:torch.Tensor):
        l1=self.dense(x)
        l2,aux=self.moe(x)
        return self.alpha*l1+(1-self.alpha)*l2,aux
