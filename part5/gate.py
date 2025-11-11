from __future__ import annotations
import torch, torch.nn as nn

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
        assert k>=1 and  k<n_expert
        self.dim=dim
        self.n_expert=n_expert
        self.k=k
        self.w_g=nn.Linear(dim,n_expert,bias=True)

    def forward(self,x:torch.Tensor):
        logits=self.w_g(x)
        prob=nn.Softmax(logits,dim=-1)
        topk,topids=torch.topk(prob,k=self.k,dim=-1)
        S=prob.size(0)
        importance=prob.mean(dim=0)
        hard_1=topk[:,0]

        load=torch.zeros(S,self.n_expert)
        load.scatter_add_(0,hard_1,torch.ones_like(hard_1,dtype=load.dtype))
        load=load/max(S,1)
        aux_loss=(self.n_expert(importance*load)).sum()
        print(importance,load,aux_loss)

