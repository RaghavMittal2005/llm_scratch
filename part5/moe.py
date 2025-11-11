from __future__ import annotations
import torch, torch.nn as nn
from gate import TopKGate
from experts import ExpertMLP

class MoE(nn.Module):
    """MixtureofExperts layer (tokenwise topk routing).
    Implementation is singleGPU friendly (loops over experts for clarity).
    https://arxiv.org/pdf/2101.03961
    """
    def __init__(self, dim: int, n_expert: int, k: int = 1, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_expert = n_expert
        self.k = k
        self.gate = TopKGate(dim, n_expert, k=k)
        self.experts = nn.ModuleList([ExpertMLP(dim, mult=mult, swiglu=swiglu, dropout=dropout) for _ in range(n_expert)])

    def forward(self,x:torch.Tensor):
        B,T,C=x.shape
        S=B*T
        x_gate=x.reshape(S,C)
        ids,w,loss=self.gate(x_gate)

        y=torch.zeros_like(x_gate)

        for e in range(self.n_expert):
            for o in range(self.k):
                mask=(ids[:,o]==0)
                if mask.any():
                    l=x_gate[mask]
                    y_e=self.experts[e](l)
                    y[mask]+=w[mask,o:o+1]*y_e
        y=y.view(B,T,C)
        return y,loss
