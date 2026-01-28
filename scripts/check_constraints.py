import json
import os
from scripts.config import get_obj_config

config = get_obj_config()

def check_violations(file_path):
    takt = config["takt"]
    drift = config["drift"]
    STATIONS = config["stations"] + 1
    
    try:
        with open(file_path, 'r') as f:
            data_list = json.load(f)
    except json.JSONDecodeError:
        return

    violations_found = False

    for entry in data_list:
        obj_id = entry.get('object')
        data = entry.get('data', {})
        offsets = entry.get('offsets', {})

        stations = [f"s{i}" for i in range(1, STATIONS)]

        for i in range(len(stations) - 1):
            current_station = stations[i]
            next_station = stations[i+1]
            
            cur_size = data.get(current_station, 0)
            cur_off = offsets.get(current_station, 0)
            next_off = offsets.get(next_station, 0)
            

            if cur_size + cur_off - takt > next_off:
                violations_found = True
                print("\n\n")
                print(f"[Violation] Object {obj_id}: {current_station} -> {next_station}")
                print(f"[Violation] {cur_size} + {cur_off} - {takt} > {next_off} (fr: {cur_size + cur_off - takt})")
                print("\n\n")

    if not violations_found:
        print("No violations found.")

if __name__ == "__main__":
    file_path = 'jsons/predicted.json' 
    check_violations(file_path)
