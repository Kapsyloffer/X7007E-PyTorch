import json
import argparse
from tqdm import tqdm
from scripts.config import get_config

config = get_config()

def overlaps(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    stations = config["stations"]
    row_height = 1
    obj_spacing = config["takt"]
    slot_width = config["drift"]
    max_x = len(data) * obj_spacing

    overlap_count = 0
    placed_rects = {y: [] for y in range(stations)}

    for obj_idx, obj in enumerate(tqdm(data)):
        start_x = obj_idx * obj_spacing

        for station_key, value in obj["data"].items():
            y = int(station_key[1:]) - 1
            offset_x = obj["offsets"].get(station_key, 0)

            x_start = start_x + offset_x
            x_end = x_start + value

            for prev_start, prev_end in placed_rects[y]:
                if x_start < prev_end and x_end > prev_start:
                    overlap_count += 1

            placed_rects[y].append((x_start, x_end))

    return overlap_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str)
    args = parser.parse_args()

    count = overlaps(args.json_file)
    print(f"{count}")
