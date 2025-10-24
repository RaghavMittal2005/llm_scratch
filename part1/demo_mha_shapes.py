import os
import math
import torch
from multihead import MultiHeadAttention
OUT_TXT = os.path.join(os.path.dirname(__file__), 'out', 'mha_shapes.txt')

def log(s):
    print(s)
    with open(OUT_TXT,'a') as f:
        f.write(s+'\n')

if __name__=="__main__":
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True) #reset file
    open(OUT_TXT, 'w').close()#reset file
    B,T,d_model,n_head=1,5,12,3
    d_head=d_model//n_head
    x=torch.randn(B,T,d_model)
    attn=MultiHeadAttention(d_model=d_model,n_head=n_head)


    qkv=attn.qkv(x)
    qkv=qkv.view(B,T,3,n_head,d_head)
    q,k,v=qkv.unbind(dim=2)
    q=q.transpose(1,2)
    k=k.transpose(1,2)
    v=v.transpose(1,2)

    score=1//math.sqrt(d_head)
    at=torch.matmul(q,k.transpose(-2,-1))*score

    att=torch.softmax(at,dim=-1)

    res=torch.matmul(att,v)
    out = res.transpose(1, 2).contiguous().view(B, T, d_model)

    

