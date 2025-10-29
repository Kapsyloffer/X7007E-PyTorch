import csv
import json
import sys
import os

takt_time = 700 #cmin
allocations = []

class Allocation:
    def __init__(self, chassi, timeslot, station, size, offset):
        self.chassi = chassi 
        self.timeslot = timeslot
        self.station = station
        self.size = size
        self.offset = offset

    def get_coords(self):
        return (self.timeslot, self.station)
    
    def get_data(self):
        return (self.size, self.offset)

    def get_id(self):
        return self.chassi;

def csv_to_json(input_file):
    data_dict = {}
    allocations.clear()  

    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            _, id_str, station_name, value, _ = row
            id_num = int(id_str)
            station_number = int(station_name.split()[-1])
            value_float = float(value)

            if id_num not in data_dict:
                data_dict[id_num] = {}

            offset = olov_offset(len(data_dict), station_number, value_float) 
            alloc = Allocation(id_num, len(data_dict), station_number, value_float, offset)
            
            data_dict[id_num][station_number] = value_float
            allocations.append(alloc)

    sorted_allocs = {}
    prev_idx = 0 
    json_list = []

    for d in data_dict:
        json_data = {}
        json_offset = {}
        json_overdraft = {}

        
        for alloc in allocations[prev_idx:]:
            if alloc.get_id() == d:
                overdraft_key = f"s{alloc.station - 1}"
                station_key = f"s{alloc.station}"
                alloc_data = alloc.get_data()
                json_data[station_key] = alloc_data[0]
                json_offset[station_key] = alloc_data[1]
                if overdraft_key != "s0":
                    json_overdraft[overdraft_key] = alloc_data[1]
                prev_idx += 1
            else:
                continue
        json_list.append({
            "id": d,
            "data": json_data,
            "offsets": json_offset,
            "overdrafts": json_overdraft
        })
    json_output = json.dumps(json_list, indent=4)
    
    out_file = os.path.splitext(input_file)[0] + ".json"
    with open(out_file, "w") as f:
        f.write(json_output)

    return json_output


    # Definitions:
    # c : chassis sequence number 1..n (allocated slot, i.e., actual position on the line)
    # s : station number 1..n
    # T : takt time (typically 700)
    # t(c,s) : operating time of c at s (sum of task times for the operator that needs most time on the station)
    #
    # d(c,s) : overdraft time of c at s (c drafts into station s+1)
    #
    # Note: all times (i.e., T, t and d) are in cmin
    #
    # Base cases:
    # d(0,s)=d(c,0)=t(0,s)=t(c,0)= 0
    #
    # Recursive (implemeted as iterative) case:
    # d(c,s) = max[0, max[d(c, s-1), d(c-1,s)] + t(c, s) - T]
def olov_offset(seq_num, station, size):
    T = takt_time #Station size / takt
    if seq_num == 0 or station == 0:
         return 0
    #d = max[0, max[d(c, s-1), d(c-1,s)] + t(c, s) - T]
    (n_l, n_r) = find_neighbours_left(seq_num, station)
    # print("\nolov data:", n_l, n_r, T)
    d = max(0, max(n_l, n_r))
    # print("new offset: ", d)
    return int(d)

def find_neighbours_left(seq_num, station):
    (neighbours_left, neighbours_right) = (-50, -50)
    for x in allocations:
        (seq, stn) = x.get_coords()
        if seq == seq_num and stn == station-1:
            (s, o) = x.get_data()
            # print("data_l: ", x.get_data())
            edge_l = int(-takt_time + s + o) #Timeslot x width, + size + offset
            edge_r = int(neighbours_right)
            (neighbours_left, neighbours_right) = (edge_l, edge_r)
        if seq == seq_num -1 and stn == station:
            (s, o) = x.get_data()
            # print("data_r: ", x.get_data())
            edge_r = int(-takt_time + s + o) #Timeslot x width, + size + offset
            edge_l = neighbours_left
            (neighbours_left, neighbours_right) = (edge_l, edge_r)

    # print("for: ", seq_num, station, "\nnl: ", neighbours_left, "nr: ", neighbours_right)
    return (neighbours_left, neighbours_right) 



if len(sys.argv) < 2:
    print("Usage: python csv_to_json.py <input_file.csv>")
    sys.exit(1)

input_path = sys.argv[1]
csv_to_json(input_path)
