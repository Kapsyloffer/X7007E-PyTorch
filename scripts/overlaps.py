import json
import argparse
import heapq
from tqdm import tqdm
from collections import defaultdict
from scripts.config import get_obj_config

config = get_obj_config()

def overlaps(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    obj_spacing = config["takt"]
    
    # 1. Pre-calculate all intervals per station
    # This is O(N)
    station_intervals = defaultdict(list)
    
    for obj_idx, obj in enumerate(data):
        start_x = obj_idx * obj_spacing
        
        for station_key, size in obj["data"].items():
            # Get integer ID (s1 -> 0, s2 -> 1, etc)
            # station_id = int(station_key[1:]) - 1 
            # Or just use the string key directly, simpler:
            
            offset_x = obj["offsets"].get(station_key, 0)
            x_start = start_x + offset_x
            x_end = x_start + size
            
            station_intervals[station_key].append((x_start, x_end))

    total_overlaps = 0

    # 2. Process each station independently
    # This is O(N log N) due to sorting, but much faster than N^2
    print("Calculating overlaps...")
    for s_key, intervals in tqdm(station_intervals.items()):
        # Sort by start time (crucial for the heap logic)
        intervals.sort(key=lambda x: x[0])
        
        # Min-heap to store 'end times' of active intervals
        active_end_times = []
        
        for start, end in intervals:
            # Clean up: Remove intervals that end before the current one starts
            # They cannot overlap with this one or any future ones (since start times increase)
            while active_end_times and active_end_times[0] <= start:
                heapq.heappop(active_end_times)
            
            # Count: Anything left in the heap MUST overlap with the current interval
            # because they started earlier (sorted input) and haven't ended yet (heap check)
            total_overlaps += len(active_end_times)
            
            # Add current interval's end time to the heap
            heapq.heappush(active_end_times, end)

    return total_overlaps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str)
    args = parser.parse_args()

    count = overlaps_optimized(args.json_file)
    print(f"Total Overlaps: {count}")
