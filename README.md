# PagedAttention (Custom KV Cache for GPT-2)

This project implements a **custom Key-Value (KV) cache manager** for transformer models, integrating with Hugging Face’s GPT-2 to enable **paged attention**. The system reduces memory overhead and allows long-sequence inference beyond the default context length.

## ✨ Features

* **Paged KV Cache** – Custom memory manager with page-based allocation and eviction.
* **GPT-2 Integration** – Replaces Hugging Face’s built-in caching with a custom attention forward pass.
* **Eviction Policies** – Sliding-window eviction to maintain bounded memory usage.
* **Binary Search Indexing** – Efficient token retrieval across pages.
* **Memory Efficiency** – Supports sequences beyond fixed limits while reducing GPU memory footprint.

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/pagedattention.git
cd pagedattention
pip install -r requirements.txt
```

## 🚀 Usage

Example: running text generation with the custom cache.

```python
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from KVStore import KVCacheManager
from GPT2_integration import PagedGPT2Attention
from generate import generate_with_paged_kv

prompt = "In a distant future, artificial intelligence"
output = generate_with_paged_kv(
    prompt,
    max_new_tokens=100,
    model_name="gpt2",
    cap=128,
    page_size=64,
)
print(output)
```

## 📂 Project Structure

```
.
├── KVStore.py             # KVCacheManager: page-based cache manager
├── GPT2_integration.py    # Custom GPT-2 attention layer with paged caching
├── generate.py            # End-to-end text generation demo
├── benchmarks/            # Benchmarking scripts & results
└── README.md
```

## 📊 Benchmarks

* Supports **400+ token sequences** with controlled eviction.
* Achieves **reduced GPU memory usage** compared to vanilla Hugging Face GPT-2 cache.
* Demonstrated correctness via token-by-token decoding with cache reuse.

## 🧠 Background

Transformers store keys and values for every token in attention. This becomes expensive for long sequences. **PagedAttention** introduces a paging mechanism that manages KV cache more like an operating system manages memory, reducing footprint and enabling scalable inference.

## 📌 Future Work

* Extend integration to other transformer models (OPT, LLaMA).
* Explore advanced eviction policies (LRU, LFU).
* Add multi-batch support.
