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

# Initialize Global State
random.seed(seed)
allocations = [[None for _ in range(stations + objects + 100)] for _ in range(stations + objects + 100)]

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
        offset = offset_left + (offset_right / 2) 
        return max(-drift_area, min(offset, drift_area))

# --- GENERATION LOGIC ---

def run_generation():
    prev_list = []

    print("Generating data...")
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

    # Helper recursive function
    def traverse_prev_recursive(alloc, idx=1, data=None, offsets=None):
        if data is None: data = {}
        if offsets is None: offsets = {}
        if alloc.prev is not None:
            idx = traverse_prev_recursive(alloc.prev, idx, data, offsets)
        key = f"s{idx}"
        data[key] = int(alloc.size)
        offsets[key] = int(alloc.offset)
        return idx + 1  

    # Helper chain function
    def chain_to_json_recursive(last_alloc, chain_id):
        data = {}
        offsets = {}
        traverse_prev_recursive(last_alloc, 1, data, offsets)
        return {
            "object": chain_id,
            "data": data,
            "offsets": offsets
        }

    # 1. Build the Master List
    master_json = []
    for chain_id, last_alloc in enumerate(prev_list, start=1):
        json_entry = chain_to_json_recursive(last_alloc, chain_id)
        master_json.append(json_entry)

    Path("jsons").mkdir(parents=True, exist_ok=True)  

    # 2. Save allocations.json (The Original)
    print("Saving jsons/allocations.json...")
    with open("jsons/allocations.json", "w") as f:
        json.dump(master_json, f, indent=4)

    # 3. Create a Copy, Shuffle it, and Save (The Puzzle)
    # We use a copy so we don't mess up the original list order in memory if we needed it later
    shuffled_json = master_json.copy()
    
    # We shuffle the list, but we DO NOT touch the IDs or Data inside.
    # Object 1 remains Object 1, even if it is now at index 99.
    random.shuffle(shuffled_json)

    print("Saving jsons/shuffled.json...")
    with open("jsons/shuffled.json", "w") as f:
        json.dump(shuffled_json, f, indent=4)

if __name__ == "__main__":
    run_generation()
