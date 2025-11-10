from .attn_modern import CausalSelfAttentionModern
from .rmsprop import RMSPROP
from .swiglu import SwiGLU
from .kv_cache import KVCache
from .rope_cache import RopeCache
import torch.nn as nn
import torch


class Transformerblock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()

        norm=RMSPROP if use_rmsnorm else nn.LayerNorm
        self.ln1=norm(n_embd)
        self.attn=CausalSelfAttentionModern(n_embd=n_embd,n_head=n_head,dropout=dropout,
                                           rope=rope,max_pos=max_pos,attention_sink=attention_sink,sliding_window=sliding_window)
        self.ln2=norm(n_embd)
        self.ffn = SwiGLU(n_embd, mult=4, dropout=dropout) if use_swiglu else nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        a,kv_cache=self.attn(self.ln1(x),kv_cache,start_pos)
        x=x+a
        x=x+self.ffn(self.ln2(x))
        return x,kv_cache