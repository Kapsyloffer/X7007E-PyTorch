import random
import json
from pathlib import Path
from config import get_obj_config
from tqdm import tqdm

config = get_obj_config()

objects = config["objects"]
stations = config["stations"]
takt = config["takt"]
drift_area = config["drift"]
gap = config["gap"]
min_size = config["min_size"] 
max_size = config["max_size"] 

TRAINING_MULTIPLIER = config["training_multiplier"] 

class Allocation:
    def __init__(self, T, S, size=takt, offset=0, prev=None):
        self.T = T
        self.S = S
        self.size = size
        self.offset = offset
        self.prev = prev
    
    def get_global_pos(self):
        start = self.T * takt + self.offset
        end = start + self.size
        return [start, end]

    def gen(self, allocation_grid):
        prevT_right = 0
        prevS_right = 0

        limit_left = self.T * takt - drift_area
        limit_right = (self.T + 1) * takt + drift_area

        if self.T > 0 and allocation_grid[self.T - 1][self.S] is not None:
            prevS_right = allocation_grid[self.T - 1][self.S].get_global_pos()[1]
        
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

        allocation_grid[self.T][self.S] = self

    def offset_calc(self, slot_left, slot_right, new_size):
            current_left = slot_left - (self.T * takt)
            target_window_size = takt - current_left
            slack = max(0, target_window_size - new_size)
            offset = current_left + (slack / 2)
            max_valid_offset = takt + drift_area - new_size
            return max(-drift_area, min(offset, max_valid_offset))


def generate_sequence(num_objects, start_id=1, random_ids=False):
    max_t = num_objects + stations + 100
    grid_w = stations + 10
    allocation_grid = [[None for _ in range(grid_w)] for _ in range(max_t)]
    
    prev_list = []

    for i in tqdm(range(0, num_objects), desc="Allocating objects", leave=False):
        prev = None
        for j in range(stations):
            T = i + j + 1
            S = j + 1
            alloc = Allocation(T, S, takt, 0, prev)
            alloc.gen(allocation_grid)  
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

    json_output = []
    for i, last_alloc in enumerate(tqdm(prev_list, desc="Processing JSON", leave=False)):
        if(random_ids):
            chain_id = random.randint(0,9999)
        else:
            chain_id = start_id + i 
        json_entry = chain_to_json_recursive(last_alloc, chain_id)
        json_output.append(json_entry)
        
    return json_output

def run_generation():
    # random.seed(seed)
    Path("jsons").mkdir(parents=True, exist_ok=True)  

    # 1. Generate the massive training list
    print(f"Generating Training Data ({objects * TRAINING_MULTIPLIER} items)...")
    training_data = generate_sequence(objects * TRAINING_MULTIPLIER, start_id=1)

    # 2. Extract a small chunk (100 items) for Overfitting Tests
    print("Saving jsons/overfitting_test.json (First 100 items)...")
    overfitting_chunk = training_data[:100]
    with open("jsons/overfitting_test.json", "w") as f:
        json.dump(overfitting_chunk, f, indent=4)

    # 3. Save the full dataset
    print("Saving jsons/allocations.json...")
    with open("jsons/allocations.json", "w") as f:
        json.dump(training_data, f, indent=4)

    print(f"Generating Test Data ({objects} items)...")
    test_data = generate_sequence(objects, start_id=1, random_ids=False)
    
    # random.seed(seed+1)
    # Shuffle the test data to create the puzzle
    random.shuffle(test_data)

    print("Saving jsons/shuffled.json...")
    with open("jsons/shuffled.json", "w") as f:
        json.dump(test_data, f, indent=4)
        
    print(f"Done. Train: {len(training_data)} items. Test: {len(test_data)} items.")

if __name__ == "__main__":
    run_generation()
