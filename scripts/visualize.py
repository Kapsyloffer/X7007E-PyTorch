import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

fig, ax = plt.subplots(figsize=(20, 12))  

colors = ["skyblue", "salmon", "gold", "lightgreen"]

slot_width = 50
max_x = len(data) * obj_spacing


overlap_count = 0
placed_rects = {y: [] for y in range(stations)}  
overlap_boxes = []  

for obj_idx, obj in enumerate(data):
    color = colors[obj_idx % len(colors)]
    start_x = obj_idx * obj_spacing

    for station_key, value in obj["data"].items():
        y = int(station_key[1:]) - 1
        offset_x = obj_spacing * int(station_key[1:]) + obj["offsets"].get(station_key, 0)

        x_start = start_x + offset_x
        x_end = x_start + value

        for prev_start, prev_end in placed_rects.get(y, []):
            if x_start < prev_end and x_end > prev_start:
                overlap_count += 1
                overlap_boxes.append((
                    max(x_start, prev_start),
                    y,
                    min(x_end, prev_end) - max(x_start, prev_start),
                    row_height
                ))

        # add rectangle to the list
        if y not in placed_rects:
            placed_rects[y] = []
        placed_rects[y].append((x_start, x_end))
        ax.add_patch(patches.Rectangle(
            (x_start, y),
            value,
            row_height,
            facecolor=color,
            edgecolor='black'
        ))

for x, y, width, height in overlap_boxes:
    if width <= 1: 
        overlap_count -= 1

    ax.add_patch(patches.Rectangle(
        (x, y),
        width,
        height,
        facecolor='red',
        alpha=0.6,
        edgecolor='black'
    ))

tick_interval = obj_spacing
ax.set_xticks(range(0, max_x + tick_interval, tick_interval))
ax.set_xticklabels([
    str(int(i / obj_spacing)) if (i // obj_spacing) % 5 == 0 else "" 
    for i in range(0, max_x + tick_interval, tick_interval)
])
ax.set_xlabel("Time")

ax.set_yticks(range(stations))
ax.set_yticklabels([f"S{i+1}" for i in range(stations)])
ax.set_ylabel("Stations")

ax.set_title(f"Assembly Line Visualization (Overlaps: {overlap_count})")
ax.invert_yaxis()
for i in range(len(data) + stations):
    x = i * obj_spacing
    ax.add_patch(patches.Rectangle(
        (x - slot_width, 0),
        2 * slot_width,
        stations,
        # facecolor='gray',
        alpha=0.5,
        edgecolor='white'
    ))
    ax.axvline(x=x, color='black', linewidth=1)


plt.show()

print(f"Total overlaps detected: {overlap_count}")

