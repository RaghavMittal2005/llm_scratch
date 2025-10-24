import torch

def attention_mask(T:torch.tensor,device=None):
    m=torch.triu(torch.ones((T,T),dtype=bool,device=device),diagonal=1)
    return m.view(1,1,T,T)