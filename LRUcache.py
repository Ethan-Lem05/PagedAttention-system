from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """Return value if key exists, else -1."""
        if key not in self.cache:
            return -1
        # Move the key to the end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Insert or update value. Evict LRU if over capacity."""
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Popitem(last=False) pops the *first* (least recently used)
            self.cache.popitem(last=False)

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        return f"LRUCache({list(self.cache.items())})"
