from __future__ import annotations
import torch
import torch.nn as nn
import math

from dataclasses import dataclass
@dataclass
class KVCache:
    k:torch.Tensor #(B,T,D,C)
    v:torch.Tensor #(B,T,D,C)

    def T(self):
        return self.k.size(2)
    
class RollingKV:
    def __init__(self, window: int, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k=k_new
            self.v=v_new
        else:
            self.k=torch.concat([self.k,k_new],dim=2)
            self.v=torch.concat([self.v,v_new],dim=2)
        
        if self.k.size(2)>self.window+self.sink:
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val, tail_v], dim=2)
        return self.k, self.v
