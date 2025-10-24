import math
import torch
import torch.nn as nn

class LearnedPosEmbeddings(nn.Module):
    def __init__(self, max_len:int,d_model:int):
        super().__init__()
        self.res=nn.Embedding(max_len,d_model)

    def forward(self,x:torch.Tensor):
        B,T,_=x.shape
        pos=torch.arange(T,device=x.device)
        emb=self.res(pos)
        return x+emb.unsqueeze(0)
    
class SinusoidalPosEmbeddings(nn.Module):
    def __init__(self, max_len:int,d_model:int):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        return x + self.pe[:T].unsqueeze(0)

