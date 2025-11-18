import random
import json

x =  700
d = 50 
t_slots = 2
stations = 5
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

        min_size = 500
        max_size = 850

        limit_left = self.T * x - d
        limit_right = (self.T + 1) * x + d

        if self.T > 0 and allocations[self.T - 1][self.S] is not None:
            prevS_right = allocations[self.T - 1][self.S].get_global_pos()[1]
        if self.prev is not None:
            prevT_right = self.prev.get_global_pos()[1]

        
        prev_left = max(prevT_right, prevS_right)
        print("prev_left: ", prev_left, "max(", prevT_right, prevS_right, ")")

        slot_left = max(prev_left, limit_left)
        print("slot_left: ", slot_left, "max(", prev_left, limit_left, ")")

        slot_right = limit_right

        available_space = slot_right - slot_left
        desired_size = random.randint(min_size, max_size)

        new_size = min(desired_size, available_space)
        new_size = max(min_size, new_size) 

        print("new size:", new_size)
        
        self.size = new_size
        print("Global pos: ", self.get_global_pos(), "\noffset&size: ", self.offset, self.size)

        #padding:
        (start, end) = self.get_global_pos()

        #Vi har start och end i en allocation,
        # Slot till höger begränsas av slot_right
        # Slot till vänster: slot_left
        # slot_right - end + start - slot_left = space

        wiggle_room = slot_right - end + start - slot_left - 2*d
        print("wiggle:", wiggle_room)

        self.offset = wiggle_room /2
        allocations[self.T][self.S] = self

        print("identifier: ", self.T - self.S, self.S, "\n\n")


def generate_json(name, shuffled):
    prev_list = []

    for i in range(0, t_slots):
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
        for new_id, entry in enumerate(json_output, start=1):
            entry["id"] = new_id 
    
    with open(f"jsons/{name}.json", "w") as f:
        json.dump(json_output, f, indent=4)

    # print(f"Sample from {name}:")
    # print(json.dumps(json_output[0], indent=4))


generate_json("allocations", False)
generate_json("shuffled", True)
