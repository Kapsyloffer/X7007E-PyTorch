import random
import json

x =  700
d = 50 
t_slots = 25
stations = 35

random.seed(seed)

allocations = [[None for _ in range(stations + t_slots)] for _ in range(stations + t_slots)]

def global_pos(T, size, offset):
    start = T * x + offset
    end = start + size
    return [start, end]

class Allocation:
    def __init__(self, T, S, size=x, offset=0, prev=None):
        self.T = T
        self.S = S
        self.size = size
        self.offset = offset
        self.prev = prev
    
    def get_global_pos(self):
        return global_pos(self.T, self.size, self.offset)


