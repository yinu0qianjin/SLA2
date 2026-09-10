
import torch
import triton
import triton.language as tl


"""
func tmp_compress_kernel

do the same thing as `compress_kernel`: compress the L dimension of X, try
to solve "too many grid num"

Max L is 524288, so idx_l can be 8191, this would cause too many grids
running simutanously. To avoid this, change it to a constant irrelavent
to idx_l.

reference:
    triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

params:
    _program_id: in range (0, 0) to (B * H, nproc)
    x: in shape [B][H][L][D]
    mean_x: in shape [B][H][(L+BLOCK_L-1)//L_BLOCKS][D]
"""
@triton.jit
def compress_kernel(
    x, mean_x,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr
):
    idx_bh = tl.program_id(0)
    nproc = tl.program_id(1)

    # the third dimension of XM
    comp_l_len = (L + BLOCK_L - 1) // BLOCK_L
    # allocate `nproc` evenly to handle `comp_l_len` blocks,
    # where comp for compressed
    # for example, nproc = 32, current proc = 3:
    # handle comp_l: 3, 32 + 3, 32 * 2 + 3, ...
    comp_l_start = nproc
    comp_l_step  = tl.num_programs(1)
    for comp_l_idx in tl.range(
        comp_l_start, comp_l_len, comp_l_step,
        # num_stages=2
    ):
        l_idx        = comp_l_idx * BLOCK_L # index of x dimension 3
        start_x      = x + idx_bh * L * D + l_idx * D
        start_mean_x = mean_x + idx_bh * comp_l_len * D + comp_l_idx * D

        # load x range: [][][BLOCK_L][D]
        range_x = tl.arange(0, BLOCK_L)[:, None] * D + tl.arange(0, D)[None, :]
        mask_x  = l_idx + tl.arange(0, BLOCK_L)[:, None] < L
        # save mean_x range: [][][1][D]
        range_mean_x = tl.arange(0, D)

        # load, compute, save
        i      = tl.load(start_x + range_x, mask=mask_x) # shape: (BLOCK_L, D)
        len_i  = min(BLOCK_L, L - l_idx)
        mean_i = tl.sum(i, axis=0) / len_i
        tl.store(start_mean_x + range_mean_x, mean_i)


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)
    # 910B1: CUBE 24, VECTOR 48

    grid = (B * H, min(L_BLOCKS, 32768 // (B * H)))
    compress_kernel[grid](x, x_mean, L, D, BLK)
    
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64, proj_q=None, proj_k=None, dtype=torch.bfloat16, stage=1):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True) # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)

    q_proj = proj_q(pooled_qblocks).to(dtype)
    k_proj = proj_k(pooled_kblocks).to(dtype)
    pooled_score = q_proj @ k_proj.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))

    if stage == 1:
        sparse_map = soft_top_k(pooled_score, topk)
        lut = None
    else:
        lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
        sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
        sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def soft_top_k(scores, k, temperature=0.1, dim=-1, max_iter=50, tol=1e-2):
    if dim != -1 and dim != scores.ndim - 1:
        scores = scores.transpose(dim, -1)

    x_min = scores.min(dim=-1, keepdim=True)[0] 
    x_max = scores.max(dim=-1, keepdim=True)[0]

    t = torch.zeros_like(x_min) 

    low = -10.0 - x_max / temperature  
    high = 10.0 - x_min / temperature   
    for it in range(max_iter):
        t_mid = (low + high) * 0.5
        logits = scores / temperature + t_mid
        probs = torch.sigmoid(logits)
        current_sum = probs.sum(dim=-1, keepdim=True)
        diff = current_sum - k
        if torch.all(torch.abs(diff) < tol):
            t = t_mid
            break
        mask = diff > 0
        low = torch.where(mask, low, t_mid)
        high = torch.where(~mask, high, t_mid)
        t = t_mid
          
    soft_logits = scores / temperature + t
    soft_mask = torch.sigmoid(soft_logits)
        
    return soft_mask
