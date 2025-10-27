from __future__ import annotations
import torch
from pathlib import Path
class ByteDataset:
    """
    block size :sequence length
    split:train &eval ratio"""
    def __init__(self,path:str,block_size:int=256,split:float=0.9):
        data=Path(path).read_bytes()
        data=torch.tensor(list(data),dtype=torch.long)
        n=int(len(data)*split)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size=block_size
        self.split=split

    def get_batch(self,which:str,batch_size:int,device=torch.device):
        buf=self.train if which =="train" else self.val
        assert len(buf)>self.block_size+1,"too small for this context window"
        idx=torch.randint(0,len(buf)-self.block_size-1,(batch_size,))
        x=torch.stack([buf[i:i+self.block_size] for i in idx])
        y=torch.stack([buf[i+1:i+1+self.block_size] for i in idx])# in the transformer model it tries to predict 
        # next token given all the previous tokens
        return x.to(device),y.to(device)

