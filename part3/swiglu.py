from __future__ import annotations
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    def __init__(self,dim:int,mult:int=4,dropout:float=0.0):
        super().__init__()
        self.w1=nn.Linear(dim,dim*mult)
        self.w2=nn.Linear(dim,dim*mult)
        self.w3=nn.Linear(dim*mult,dim)
        self.act=nn.SiLU()
        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x1=self.w1(x)
        x2=self.w2(x)
        x=self.act(x2)*x1
        x=self.w3(x)
        return self.dropout(x)

