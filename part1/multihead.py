import os
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from causal_mask import attention_mask

class MultiHeadAttention(nn.Module):

    """1.4 Multi-head attention with explicit shape tracing.

    Dimensions (before masking):
      x:      (B, T, d_model)
      qkv:    (B, T, 3*d_model)
      view→   (B, T, 3, n_head, d_head)   where d_head = d_model // n_head
      split→  q,k,v each (B, T, n_head, d_head)
      swap→   (B, n_head, T, d_head)
      scores: (B, n_head, T, T) = q @ k^T / sqrt(d_head)
      weights:(B, n_head, T, T) = softmax(scores)
      ctx:    (B, n_head, T, d_head) = weights @ v
      merge:  (B, T, n_head*d_head) = (B, T, d_model)
    """
    def __init__(self,d_model:int,n_head:int,dropout:float=0.0, trace_shapes:bool=True):
        super.__init__()
        assert d_model%n_head==0,"d_model must be divisible by n_head"  #must be satisfied
        d_model=self.d_model
        n_head=self.n_head
        d_head=d_model//n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self,x:torch.tensor):
        B,T,C=x.shapes
        B, T, C = x.shape
        qkv = self.qkv(x)                          # (B,T,3*C)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head) #similar to reshape but is faster and requires contigous allocation
        if self.trace_shapes:
            print("qkv view:", qkv.shape)
        q, k, v = qkv.unbind(dim=2) # we unbind it from 3rd dimension b,t,3(split)
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)
        if self.trace_shapes:
            print("q view:", q.shape,"k view:",k.shape,"v view:",v.shape)
        score=1.0/math.sqrt(self.d_head)
        attn=torch.matmul(q,k.transpose(-2,-1))*score
        mask=attention_mask(T=T,device=x.device)
        attn=attn.masked_fill(mask,float('inf'))
        w=F.softmax(attn,dim=-1)
        w=self.dropout(w)
        ctx=torch.matmul(w,v)
        out = ctx.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,d_model)
        out = self.proj(out)
        return out,w

        

        