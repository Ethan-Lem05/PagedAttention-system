import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from KVStore import KVCacheManager
from copy import deepcopy
from GPT2_integration import PagedGPT2Attention

def generate_with_paged_kv(
    prompt: str,
    max_new_tokens: int = 50,
    model_name: str = "gpt2",
    cap: int = 128,
    page_size: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = None,
    device: str | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    #if device is None:
    #    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.use_cache = False  # custom cache management
    model.eval().to(device)

    cache = KVCacheManager(cap=cap, page_size=page_size)
    cache.next_token_idx = 0

    for i, block in enumerate(model.transformer.h):
        old = block.attn
        new = PagedGPT2Attention(
            model.config,
            is_cross_attention=False,   # IMPORTANT
            layer_idx=i,
            cache_manager=cache
        )
        with torch.no_grad():
            new.c_attn.weight.copy_(old.c_attn.weight)
            new.c_attn.bias.copy_(old.c_attn.bias)
            new.c_proj.weight.copy_(old.c_proj.weight)
            new.c_proj.bias.copy_(old.c_proj.bias)
        new.resid_dropout = old.resid_dropout
        block.attn = new
    
    # helper functions
    def reset_cache(cm: KVCacheManager):
        cm.cache.od.clear()
        cm.global_id = 0
        cm.current_page_id = None
        cm.evictions = 0
        cm.next_token_idx = 0

    def sample_from_logits(logits_row: torch.Tensor) -> int:
        """logits_row: [vocab]. Applies temperature + top-p nucleus sampling."""
        if temperature <= 0:
            # degenerate -> greedy
            return int(torch.argmax(logits_row).item())

        logits = logits_row / temperature
        probs = torch.softmax(logits, dim=-1)

        # top-p nucleus filter
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cdf > top_p).nonzero(as_tuple=False)
            if cutoff.numel() > 0:
                first_cut = cutoff[0, 0]
                keep_idx = sorted_idx[: first_cut + 1]
            else:
                keep_idx = sorted_idx
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[keep_idx] = True
            probs = torch.where(mask, probs, torch.zeros_like(probs))
            probs = probs / probs.sum()  # renormalize

        next_id = torch.multinomial(probs, num_samples=1)
        return int(next_id.item())
    
    reset_cache(cache)
    inp = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model(**inp)  # runs full prompt through, populates cache

    generated = inp["input_ids"]

    # 6) Autoregressive decode
    with torch.no_grad():
        for _ in range(max_new_tokens):
            last_token = generated[:, -1:]  # T=1
            out = model(input_ids=last_token)
            logits = out.logits[:, -1, :]  # [B, vocab], B=1

            # choose next token
            if temperature == 1.0 and top_p == 1.0:
                # greedy fast-path
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                nid = sample_from_logits(logits[0])
                next_id = torch.tensor([[nid]], device=device, dtype=torch.long)

            generated = torch.cat([generated, next_id], dim=1)

    text = tok.decode(generated[0], skip_special_tokens=True)

    # You can inspect cache.evictions if you want
    # print("Pages evicted:", cache.evictions)

    return text, generated, cache, model, tok

if __name__ == "__main__":
    out_text, ids, cache, model, tok = generate_with_paged_kv(
        prompt="In a distant future,",
        max_new_tokens=60,
        cap=256,
        page_size=64,
        temperature=0.9,
        top_p=0.9,
        seed=42,
    )
    print(out_text)
    print("Evictions:", cache.evictions)