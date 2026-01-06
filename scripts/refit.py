import sys
import os
import json

takt_time = 700 #cmin
allocations = []

class Allocation:
    def __init__(self, period, chassi, timeslot, station, size, offset):
        self.period = period
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

    def get_period(self):
        return self.period

def load_and_process_allocations(input_file):
    
    with open(input_file, 'r') as f:
        json_data = json.load(f)

    for entry in json_data:
        # Extract object-level data
        chassi = entry.get("object")
        # Use id if present, otherwise default to object (sequence)
        timeslot = entry.get("id", chassi)
        
        # Period is required by Allocation class but missing from the JSON snippet
        # Defaulting to "Unknown" or checking if key exists
        period = entry.get("period", "Unknown") 
        
        data_dict = entry.get("data", {})
        
        # Ensure offsets dict exists to save results
        if "offsets" not in entry:
            entry["offsets"] = {}

        # Iterate through the stations in the data dictionary
        for station_key, size_value in data_dict.items():
            
            # Key format is "s1", "s2", etc. Extract integer station number.
            if station_key.startswith("s"):
                try:
                    station_number = int(station_key.replace("s", ""))
                except ValueError:
                    continue # Skip invalid station keys
            else:
                continue

            # Calculate offset using olov_offset instead of reading from JSON
            offset_value = olov_offset(int(timeslot), station_number, float(size_value))
            
            # Save calculated offset back to the JSON structure
            entry["offsets"][station_key] = offset_value

            # Create the Allocation object
            # Structure: period, chassi, timeslot, station, size, offset
            alloc = Allocation(
                period, 
                int(chassi), 
                int(timeslot), 
                station_number, 
                float(size_value), 
                float(offset_value)
            )
            
            allocations.append(alloc)
            
    return json_data

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
    print("\nolov data:", n_l, n_r, T)
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

def save_json(output_file, data):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved changes to {output_file}")

if len(sys.argv) < 2:
    print("Usage: python refit.py <input_file.json>")
    sys.exit(1)

input_path = sys.argv[1]
processed_data = load_and_process_allocations(input_path)
save_json(input_path, processed_data)
