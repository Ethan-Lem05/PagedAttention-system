from collections import deque
from LRUcache import LRUCache
import torch

class PageRef:
    def __init__(self, ref_id: int, start_token_idx: int, page_size: int):
        self.pageRefID = ref_id
        self.range: tuple[int, int] = (start_token_idx, start_token_idx + page_size)
        self.cap = page_size
        self.tensors: list[tuple[torch.Tensor, torch.Tensor]] = []

    def is_full(self):
        return len(self.tensors) >= self.cap

    def write_page(self, k: torch.Tensor, v: torch.Tensor) -> int:
        if self.is_full():
            return -1
        self.tensors.append((k, v))
        return len(self.tensors) - 1

    def read_token(self, token_idx: int):
        lo, hi = self.range
        if lo <= token_idx < hi:
            local_idx = token_idx - lo
            if local_idx < len(self.tensors):
                return self.tensors[local_idx]
        return None

    def read_page(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.tensors


class KVCacheManager:
    def __init__(self, cap: int, page_size: int):
        self.cache = LRUCache(capacity=cap)
        self.cap = cap
        self.page_size = page_size
        self.global_id = 0
        self.current_page_id = None

        # VISIBILITY
        self.evictions = 0

    def _allocate_new_page(self, start_token_idx):
        page = PageRef(self.global_id, start_token_idx, self.page_size)
        pid = self.global_id
        self.global_id += 1
        ev = self.cache.put(pid, page)
        if ev is not None:
            # Optional: free GPU/CPU buffers in ev[1] if needed
            self.evictions += 1
        self.current_page_id = pid
        return page

    def _current_page(self):
        if self.current_page_id is None:
            return None
        page = self.cache.get(self.current_page_id)  # marks MRU
        # if it was evicted for some reason, page could be None
        return page

    def append_KV_pair(self, token_idx: int, k: torch.Tensor, v: torch.Tensor):
        page = self._current_page()
        if page is None or page.is_full():
            page = self._allocate_new_page(token_idx)

        page.write_page(k, v)

        # Touch in LRU to mark this page as recently used
        ev = self.cache.put(self.current_page_id, page)

    def read_token(self, token_idx: int):
        for pid, page in self.cache.od.items():
            start, end = page.range
            if start <= token_idx < end:
                page = self.cache.get(pid) # bump recency
                return page.read_token(token_idx)
        return None
    
    def gather(self, seq_len):
        items = list(self.cache.od.items())
        items.sort(key=lambda kv: kv[1].range[0]) # sort pages oldest to newest
        k_all, v_all = [], []

        for pid, _ in items:  # for all pages from oldest to newest
            page = self.cache.get(pid)
            for k, v in page.tensors: # iterate through pages
                k_all.append(k) 
                v_all.append(v)

        if not k_all or not v_all: # torch.stack does not work if there are any None tensors
            return None, None
        
        return torch.stack(k_all, dim=0), torch.stack(v_all, dim=0)

