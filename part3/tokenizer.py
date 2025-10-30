from __future__ import annotations
import torch

class ByteTokenizer:
    def encode(self,s:str)->torch.Tensor:
        return torch.Tensor(list(s.encode("utf-8")),dtype=torch.long)
    def decode(self,ids:torch.Tensor)->str:
        if isinstance(ids,torch.Tensor):
            ids=ids.tolist()
        return bytes(ids).decode("utf-8",errors="ignore")
    @property
    def vocab_size(self)->int:
        return 256