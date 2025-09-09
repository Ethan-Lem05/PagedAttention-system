from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.od = OrderedDict()

    def get(self, key):
        if key not in self.od:
            return None
        self.od.move_to_end(key)  # becomes most recently used
        return self.od[key]

    def put(self, key, value):

        evicted = None

        if key in self.od: # if updated then becomes most recently used
            self.od.move_to_end(key)
        self.od[key] = value

        if len(self.od) > self.capacity:
            evicted = self.od.popitem(last=False)  
        return evicted # returns the evicted page or None if no pages evicted 

    def __contains__(self, key): 
        return key in self.od
    
    def __len__(self): 
        return len(self.od)
