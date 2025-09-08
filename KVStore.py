from collections import deque
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
    def __init__(self, cap):
        self.cache = deque()
        self.cap = cap
        self.page_size = 4
        self.global_id = 0

        # VISIBILITY
        self.evictions = 0

    def append_KV_pair(self, token_idx: int, k: torch.Tensor, v: torch.Tensor):
        if not self.cache or self.cache[0].is_full():
            page = PageRef(self.global_id, token_idx, self.page_size)
            self.global_id += 1
            self.cache.appendleft(page)
        else:
            page = self.cache[0]

        page.write_page(k, v)

        if len(self.cache) > self.cap:
            self.cache.pop()
            self.evictions += 1

    def read_token(self, token_idx: int):
        for page in self.cache:
            if page.range[0] <= token_idx < page.range[1]:
                return page.read_token(token_idx)
        return None
    
    def gather(self, seq_len: int):
        k_all, v_all = [], []
        # Only search within cached token ranges
        for page in reversed(self.cache):  # oldest → newest
            for (k, v) in page.tensors:
                k_all.append(k)
                v_all.append(v)
        if not k_all or not v_all:
            return None, None
        return torch.stack(k_all, dim=0), torch.stack(v_all, dim=0)

