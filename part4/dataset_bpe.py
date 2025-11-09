from __future__ import annotations
import torch
from pathlib import Path
from typing import Tuple
from tokeniser_bpe import BPETokenizer
from torch.utils.data import Dataset

class BPEByteDataset(Dataset):
    def __init__(self, path: str, tokenizer: BPETokenizer, block_size: int = 256):
        super().__init__()
        self.block_Size=block_size
        data=Path(path).read_text(encoding="utf-8")
        self.ids=torch.Tensor(tokenizer.encode(data),dtype=torch.long)

    def __len__(self):
        return len(self.ids)-(self.block_Size+1)
    def __getitem__(self,i:int)->Tuple[torch.Tensor,torch.Tensor]:
        x=self.ids[i:self.block_Size+i]
        y=self.ids[i+1:self.block_Size+i+1]
        return x,y
def make_loader(path:str,tokeniser:BPETokenizer,batch_Size:int,block_size:int=256):

    ds=BPEByteDataset(path,tokeniser,block_size)
    return torch.utils.data.DataLoader(ds,batch_size=batch_Size,shuffle=True,drop_last=True)
