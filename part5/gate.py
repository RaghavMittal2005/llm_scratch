from __future__ import annotations
import torch, torch.nn as nn
import torch.nn.functional as F
class TopKGate(nn.Module):
    """Topk softmax gating with Switch-style loadbalancing aux loss.
    Args:
      dim: input hidden size
      n_expert: number of experts
      k: number of experts to route per token (1 or 2 typical)
    Returns:
      (indices, weights, aux_loss) where
        indices: (S, k) long, expert ids for each token
        weights: (S, k) float, gate weights (sum ≤ 1 per token)
        aux_loss: scalar load-balancing penalty
    """
    def __init__(self,dim:int,n_expert:int,k:int=1):
        super().__init__()
        assert k>=1 and  k<=n_expert
        self.dim=dim
        self.n_expert=n_expert
        self.k=k
        self.w_g=nn.Linear(dim,n_expert,bias=True)

    def forward(self,x:torch.Tensor):
        logits=self.w_g(x)
        prob=torch.softmax(logits,dim=-1)
        topk,topids=torch.topk(prob,k=self.k,dim=-1)
        S=prob.size(0)
        E=prob.size(1)
        importance=prob.mean(dim=0)
        hard_1=topids[:,0]
        normalized_weights=topk/topk.sum(dim=-1,keepdim=True)
        load=torch.zeros(E,device=x.device)
        load.scatter_add_(0,hard_1,torch.ones_like(hard_1,dtype=load.dtype))
        load=load/S
        aux_loss=(E*(importance*load)).sum()
        print(importance,load,aux_loss)
        return topids,normalized_weights,aux_loss

