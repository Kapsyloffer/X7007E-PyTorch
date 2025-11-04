import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("json_file", type=str)
args = parser.parse_args()

with open(args.json_file, "r") as f:
    data = json.load(f)

if isinstance(data, dict):
    data = list(data.values())

stations = 38
row_height = 1
obj_spacing = 700
slot_width = 50
max_x = len(data) * obj_spacing

overlap_count = 0
placed_rects = {y: [] for y in range(stations)}

for obj_idx, obj in enumerate(data):
    start_x = obj_idx * obj_spacing

    for station_key, value in obj["data"].items():
        y = int(station_key[1:]) - 1
        offset_x = obj["offsets"].get(station_key, 0)

        # Each object’s x range
        x_start = start_x + offset_x
        x_end = x_start + value

        # Compare with all previously placed rectangles at this station
        for prev_start, prev_end in placed_rects[y]:
            if x_start < prev_end and x_end > prev_start:
                overlap_count += 1

        # Record this rectangle so future ones can check against it
        placed_rects[y].append((x_start, x_end))

print(f"Total overlaps detected: {overlap_count}")
