
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from kernel import _attention
from utils import get_block_map


class SparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk, L, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True, layer_idx=None, mode="infer", stage=1, router_data_path=None):
        R'''
        Args:
            head_dim: dimension of each head has been removed, since the initialization of Router and alpha has been moved out
            topk: ratio of keys selected for sparse attention, shared across all queries.
            L: int, seqlenth
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
            layer_idx: int, if mode == "train" and stage == 2, the parameter must provide.
            mode: str, "train" or "infer".
            stage: int, 1 or 2, training stage.
            router_data_path: str, if mode=="train" and stage == 2, the parameter must provide.
        '''
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.layer_idx = layer_idx
        self.router_data_path = router_data_path
        self.stage = stage

        self.proj_q = nn.Linear(head_dim, head_dim, dtype=torch.float32)
        self.proj_k = nn.Linear(head_dim, head_dim, dtype=torch.float32)
        
        num_blocks = (L + self.BLKQ -1) // self.BLKQ
        self.alpha = nn.Parameter(torch.full((num_blocks, 1), 0.8), requires_grad=True)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        if mode == "train":
            if self.stage == 2:
                self.init_weights_2_()
                self.proj_q.weight.requires_grad = False
                self.proj_q.bias.requires_grad = False
                self.proj_k.weight.requires_grad = False
                self.proj_k.bias.requires_grad = False
            else:
                self.init_weights_1_()

    def init_weights_1_(self):
        with torch.no_grad():
            nn.init.eye_(self.proj_q.weight)
            nn.init.eye_(self.proj_k.weight)
            nn.init.zeros_(self.proj_q.bias)
            nn.init.zeros_(self.proj_k.bias)
        
    def init_weights_2_(self):
        data = torch.load(self.router_data_path + f"/block{self.layer_idx}/block_{self.layer_idx}_2.pt")
        with torch.no_grad():
            self.proj_q.weight.data = data['proj_q.weight']
            self.proj_q.bias.data = data['proj_q.bias']
            self.proj_k.weight.data = data['proj_k.weight']
            self.proj_k.bias.data = data['proj_k.bias']
            self.alpha.data = data['alpha']


    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
        '''
        B, num_heads, L, head_dim = q.size()

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK, proj_q=self.proj_q, proj_k=self.proj_k, dtype=self.dtype, stage=self.stage)
        

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        if self.stage == 1:
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            soft_mask_ = torch.repeat_interleave(torch.repeat_interleave(sparse_map, 64, dim=2), 64, dim=3)
            attention_weights = F.softmax(scores * soft_mask_[:, :, :L, :L], dim=-1)
            o_s = torch.matmul(attention_weights, v)
        else:
            o_s = _attention.apply(q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)
        

        q = self.feature_map_q(q).contiguous().to(self.dtype) # c_q
        k = self.feature_map_k(k).contiguous().to(self.dtype) # c_k
        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))
        o_l = calc_linear(q, k, v)

        block_indices = torch.arange(L, device=q.device) // self.BLKQ
        alpha_per_position = self.alpha[block_indices].view(1, 1, L, 1)

        weighted_o_s = alpha_per_position * o_s
        weighted_o_l = (1 - alpha_per_position) * o_l

        o = (weighted_o_s + weighted_o_l).to(self.dtype)
        return o
