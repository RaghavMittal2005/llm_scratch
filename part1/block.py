from ffn import FeedForwardNetwork
from multihead import MultiHeadAttention
import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,n_head:int,dropout:int=0):
        super.__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=MultiHeadAttention(d_model,n_head,dropout)
        self.ln2=nn.LayerNorm(d_model)
        self.ffn=FeedForwardNetwork(d_model,4,dropout)

    def forward(self,x:torch.Tensor):
        x=x+self.attn(self.ln1(x))[0]
        x=x+self.ffn(self.ln2(x))
        return x
