import torch
from __future__ import annotations

class ByteTokeniser:
    """
    uses simple utf-8 encoding byte wise for encoding
    with vocab size 256 that is the unique number of values it has like A has 65
    """
    def encode(self,s:str)->torch.tensor:
        return torch.tensor(list(s.encode(encoding="utf-8",errors="ignore")),dtype=torch.long)
    
    def decode(self,ids:torch.tensor)->str:
        if isinstance(ids,torch.Tensor):
            ids=ids.tolist()

        return bytes(ids).decode(encoding="utf-8",errors="ignore")
    
    @property
    def vocab_size(self)->int:
        return 256
    