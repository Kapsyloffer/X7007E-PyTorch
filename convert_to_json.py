import csv
import json
import sys
import os

takt_time = 700 #cmin
allocations = []

class Allocation:
    def __init__(self, timeslot, station, size, offset):
        self.timeslot = timeslot
        self.station = station
        self.size = size
        self.offset = offset

    def get_coords(self):
        return (self.timeslot, self.station)
    
    def get_data(self):
        return (self.size, self.offset)

def csv_to_json(input_file):
    """Convert CSV to JSON with local offsets for each station."""
    data_dict = {}

    # Read CSV and group by ID
    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            _, id_str, station_name, value, _ = row
            id_num = int(id_str)
            station_number = int(station_name.split()[-1])
            value_float = float(value)

            if id_num not in data_dict:
                data_dict[id_num] = {}
            data_dict[id_num][station_number] = value_float
            allocations[id_num][station_number]

    output_list = []

    # Convert to JSON structure with local offsets
    for id_num, stations in data_dict.items():
        stations_sorted = sorted(stations.keys())
        data_json = {f"s{s}": round(stations[s]) for s in stations_sorted}
        offsets_json = {f"s{s}": -max(0, round(stations[s] - takt_time)) for s in stations_sorted}

        output_list.append({ "id": id_num, "data": data_json,
            "offsets": offsets_json
        })

    # Write JSON
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}.json"
    with open(output_file, "w") as jsonfile:
        json.dump(output_list, jsonfile, indent=4)
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
    T = x #Station size / takt
    if seq_num == 0 or station == 0:
        return 0
    #d = max[0, max[d(c, s-1), d(c-1,s)] + t(c, s) - T]
    if allocations[seq_num][station-1] is None or allocations[seq_num-1][station] is None:
        return 0
    d = max(0, max(allocations[seq_num][station-1].get_offset(), allocations[seq_num-1][station].get_offset()) + size - T)
    allocations[seq_num][station].set_offset(d)
    print("new offset: ", d)
    return d


if len(sys.argv) < 2:
    print("Usage: python csv_to_json.py <input_file.csv>")
    sys.exit(1)

input_path = sys.argv[1]
csv_to_json(input_path)
