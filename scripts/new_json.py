import random
import json
from pathlib import Path
from config import get_config

config = get_config()

objects = config["objects"]
stations = config["stations"]

takt = config["takt"]
drift_area = config["drift"]
gap = config["gap"]

seed = 1337      

min_size = 300
max_size = takt + 2 * drift_area

random.seed(seed)

allocations = [[None for _ in range(stations + objects)] for _ in range(stations + objects)]

def global_pos(T, size, offset):
    start = T * takt + offset
    end = start + size
    return [start, end]

class Allocation:
    def __init__(self, T, S, size=takt, offset=0, prev=None):
        self.T = T
        self.S = S
        self.size = size
        self.offset = offset
        self.prev = prev
    
    def get_global_pos(self):
        return global_pos(self.T, self.size, self.offset)

    def get_prev(self):
        return prev

    def gen(self):
        prevT_right = 0
        prevS_right = 0

        limit_left = self.T * takt - drift_area
        limit_right = (self.T + 1) * takt + drift_area

        if self.T > 0 and allocations[self.T - 1][self.S] is not None:
            prevS_right = allocations[self.T - 1][self.S].get_global_pos()[1]
        if self.prev is not None:
            prevT_right = self.prev.get_global_pos()[1]

        
        prev_left = max(prevT_right, prevS_right) 
        slot_left = max(prev_left, limit_left) + gap
        slot_right = limit_right - gap
        available_space = slot_right - slot_left 

        desired_size = random.randint(min_size, max_size)
        new_size = min(desired_size, available_space)
        new_size = max(min_size, new_size) 
      
        self.offset = self.offset_calc(slot_left, slot_right, new_size)
        
        self.size = new_size

        allocations[self.T][self.S] = self

    def offset_calc(self, slot_left, slot_right, new_size):
        offset_left = slot_left - self.T * takt 
        offset_right = takt + drift_area - offset_left - new_size
        offset = offset_left + (offset_right / 2) #utilize space between

        return max(-drift_area, min(offset, drift_area)) # -d < offset < d



def generate_json(name, shuffled):
    prev_list = []

    for i in range(0, objects):
        prev = None
        for j in range(stations):
            T = i + j + 1
            S = j + 1
            alloc = Allocation(T, S, takt, 0, prev)
            alloc.gen()
            allocations[T][S] = alloc  
            prev = alloc
            if j == stations - 1:
                prev_list.append(prev)

    def traverse_prev_recursive(alloc, idx=1, data=None, offsets=None):
        if data is None:
            data = {}
        if offsets is None:
            offsets = {}

        if alloc.prev is not None:
            idx = traverse_prev_recursive(alloc.prev, idx, data, offsets)

        key = f"s{idx}"
        data[key] = int(alloc.size)
        offsets[key] = int(alloc.offset)
        return idx + 1  

    def chain_to_json_recursive(last_alloc, chain_id):
        data = {}
        offsets = {}
        traverse_prev_recursive(last_alloc, 1, data, offsets)
        return {
            "object": chain_id,
            "data": data,
            "offsets": offsets
        }

    json_output = []
    for chain_id, last_alloc in enumerate(prev_list, start=1):
        json_entry = chain_to_json_recursive(last_alloc, chain_id)
        json_output.append(json_entry)
    
    if shuffled:
        random.shuffle(json_output)
        for new_id, entry in enumerate(json_output, start=1):
            entry["id"] = new_id 
   
    Path("jsons").mkdir(parents=True, exist_ok=True)  
    with open(f"jsons/{name}.json", "w") as f:
        json.dump(json_output, f, indent=4)

    # print(f"Sample from {name}:")
    # print(json.dumps(json_output[0], indent=4))


generate_json("allocations", False)
generate_json("shuffled", True)
