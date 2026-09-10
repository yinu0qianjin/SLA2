import torch
import torch_npu
from core import SparseLinearAttention



def test_stage1():
    module = SparseLinearAttention(head_dim=128, topk=0.05, L=32760, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True, mode="train", stage=1)
    module = module.npu()
    
    B, L, H, D = 1, 32760, 2, 128
    dtype = torch.bfloat16
    device = "npu"
    q = torch.randn((B, H, L, D), dtype=dtype, device=device)
    k = torch.randn((B, H, L, D), dtype=dtype, device=device)
    v = torch.randn((B, H, L, D), dtype=dtype, device=device)
    o = module(q, k, v)
    print(o.shape)
    do = torch.randn((B, H, L, D), dtype=torch.bfloat16).npu()
    o.backward(do)
    

def test_stage2():
    module = SparseLinearAttention(head_dim=128, topk=0.05, L=32760, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True, mode="train", stage=2, router_data_path="../../stage1/0603/checkpoints_block_97/", layer_idx=0)
    module = module.npu()
    
    B, L, H, D = 1, 32760, 2, 128
    dtype = torch.bfloat16
    device = "npu"
    q = torch.randn((B, H, L, D), dtype=dtype, device=device)
    k = torch.randn((B, H, L, D), dtype=dtype, device=device)
    v = torch.randn((B, H, L, D), dtype=dtype, device=device)
    o = module(q, k, v)
    print(o.shape)
    do = torch.randn((B, H, L, D), dtype=torch.bfloat16).npu()
    o.backward(do)



if __name__ == "__main__":
    test_stage1()
    test_stage2()
