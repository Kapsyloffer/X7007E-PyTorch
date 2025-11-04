import random
import json

x =  700
d = 50 
t_slots = 3000
stations = 38  
seed = 1337      

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

    def gen(self):
        prevT_right = 0
        prevS_right = 0

        limit_left = self.T * x 
        limit_right = (self.T + 1) * x + 2 * d

        if self.T > 0 and allocations[self.T - 1][self.S] is not None:
            prevS_right = allocations[self.T - 1][self.S].get_global_pos()[1]
        if self.prev is not None:
            prevT_right = self.prev.get_global_pos()[1]
        
        slot_left = max(limit_left, max(prevT_right, prevS_right))
        slot_right = limit_right

        gap = random.randint(0, 2*d // 2)

        available_space = slot_right - slot_left + gap
        new_size = min(x - gap, available_space)
        new_size = max(50, new_size)

        self.offset = (slot_left + gap) - (self.T * x)
        self.size = new_size

        allocations[self.T][self.S] = self


def generate_json(name, shuffled):
    prev_list = []

    for i in range(t_slots):
        prev = None
        for j in range(stations):
            T = i + j + 1
            S = j + 1
            alloc = Allocation(T, S, x, 0, prev)
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
        data[key] = alloc.size
        offsets[key] = alloc.offset
        return idx + 1  

    def chain_to_json_recursive(last_alloc, chain_id):
        data = {}
        offsets = {}
        traverse_prev_recursive(last_alloc, 1, data, offsets)
        return {
            "id": chain_id,
            "data": data,
            "offsets": offsets
        }

    json_output = []
    for chain_id, last_alloc in enumerate(prev_list, start=1):
        json_entry = chain_to_json_recursive(last_alloc, chain_id)
        json_output.append(json_entry)
    
    if shuffled:
        random.shuffle(json_output)

        for entry in json_output:
            
            for key in entry["offsets"]:
                jitter = random.randint(0, 5*d) 
                entry["offsets"][key] += int(jitter)

            for key in entry["data"]:
                entry["data"][key] = int(entry["data"][key] * random.uniform(0.8, 1.2))

        for new_id, entry in enumerate(json_output, start=1):
            entry["id"] = new_id 
    
    with open(f"jsons/{name}.json", "w") as f:
        json.dump(json_output, f, indent=4)

    # print(f"Sample from {name}:")
    # print(json.dumps(json_output[0], indent=4))


generate_json("allocations", False)
generate_json("shuffled", True)
