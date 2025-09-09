from KVStore import KVCacheManager
import torch 
import torch.nn.functional as F


def decode_step(cache: KVCacheManager, query: torch.Tensor, seq_len: int):

    k_all, v_all = cache.gather(seq_len)

    q = query.transpose(0, 1)                # [n_heads, 1, head_dim]
    k = k_all.transpose(0, 1)              # [n_heads, seq_len, head_dim]
    v = v_all.transpose(0, 1)

    out = F.scaled_dot_product_attention(q,k,v, is_causal=True)

    return out.transpose(0,1)

if __name__ == "__main__":
    n_heads, head_dim = 2, 4
    cache = KVCacheManager(cap=5, page_size=10)

    # Append 6 tokens
    for i in range(400):
        k = torch.randn(n_heads, head_dim)
        v = torch.randn(n_heads, head_dim)
        cache.append_KV_pair(i, k, v)

    # Query for the "next token"
    q_t = torch.randn(1, n_heads, head_dim)

    # Decode
    out = decode_step(cache, q_t, seq_len=400)
    print("Attention output:", out)
    print(cache.evictions)