import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from KVStore import KVCacheManager  # your class

class PagedGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, cache_manager=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.cache_manager = cache_manager
        self.next_token_idx = 0

    def forward(self, hidden_states, layer_past=None, **kwargs):
        # gather the dimensions well need
        B, T, _ = hidden_states.size()
        if B != 1:
            raise ValueError("This MVP integration supports batch=1 only.")
        
        # compute K V and Q for the current hidden state 
        qkv = self.c_attn(hidden_states)  # [B, T, 3*E]
        q, k, v = qkv.split(self.split_size, dim=2)  # each [B, T, E]

        def _shape(x):
            return x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # align the shapes

        q = _shape(q)  # [1, H, T, D]
        k = _shape(k)  # [1, H, T, D]
        v = _shape(v)  # [1, H, T, D]

        # append K and V to the KV cache 
        for t in range(T):
            k_tok = k[0, :, t, :].contiguous()  # [H, D]
            v_tok = v[0, :, t, :].contiguous()  # [H, D]
            self.cache_manager.append_KV_pair(self.next_token_idx, k_tok, v_tok)
            self.next_token_idx += 1

        # use gather() to read all K and V vectors into matricies 
        k_all, v_all = self.cache_manager.gather(self.cache_manager.next_token_idx)

        # Fallback (shouldn't happen since we just appended), but keep it safe
        if k_all is None or v_all is None:
            # Build from current mini-batch only
            # k[0] is [H, T, D] -> transpose to [T, H, D]
            k_all = k[0].permute(1, 0, 2).contiguous()
            v_all = v[0].permute(1, 0, 2).contiguous()

        # use Q to compute embeddings 
        import torch.nn.functional as F
        q_heads = q[0]                       # [H, T, D]
        k_heads = k_all.transpose(0, 1)      # [H, S, D]
        v_heads = v_all.transpose(0, 1)      # [H, S, D]


        attn_out = F.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, is_causal=True
        )  # [H, T, D]


        # align tensors with GPT-2
        context = attn_out.permute(1, 0, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        attn_output = self.c_proj(context)
        attn_output = self.resid_dropout(attn_output)

        # return those tensors
        present = None
        return attn_output, present