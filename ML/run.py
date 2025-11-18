import torch
from pathlib import Path
import json
from model import build_transformer
from dataset import ItemReorderingDataset
import torch.nn as nn
from tqdm import tqdm

takt = 700

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

CONFIG = {
    "json_path": "jsons/predicted.json",
    "model_folder": "weights",
    "d_model": 256,
    "epoch": 9,
    "output_json": "jsons/predicted.json"
}

class NumericInputWrapper(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.proj(x)

def load_model(device, dataset):
    num_stations = dataset[0][0].shape[0]
    input_dim = dataset[0][0].shape[1]

    model = build_transformer(
        src_vocab_size=num_stations,
        tgt_vocab_size=num_stations,
        src_seq_len=num_stations,
        tgt_seq_len=num_stations,
        d_model=CONFIG["d_model"]
    ).to(device)

    model.src_embed = NumericInputWrapper(input_dim=input_dim, d_model=CONFIG["d_model"]).to(device)
    model.tgt_embed = NumericInputWrapper(input_dim=input_dim, d_model=CONFIG["d_model"]).to(device)

    model_path = Path(CONFIG["model_folder"]) / f"epoch_{CONFIG['epoch']:02d}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_sample_score(model, sample_tensor, device):
    sample_tensor = sample_tensor.unsqueeze(0).to(device)
    tgt_tensor = sample_tensor

    with torch.no_grad():
        enc_out = model.encode(sample_tensor.float(), src_mask=None)
        dec_out = model.decode(enc_out, src_mask=None, tgt=tgt_tensor.float(), tgt_mask=None)
        logits = model.project(dec_out)
        score = logits.mean().item()
    return score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load JSON
    with open(CONFIG["json_path"], "r") as f:
        raw_json = json.load(f)

    model = load_model(device, ItemReorderingDataset(CONFIG["json_path"]))

    # Score every JSON entry
    sample_scores = []
    for idx, obj in tqdm(enumerate(raw_json), desc="Scoring samples", total=len(raw_json)):
        data_tensor = torch.tensor([[size, 0] for size in obj["data"].values()])  # offsets ignored
        score = predict_sample_score(model, data_tensor, device)

        sample_scores.append((idx, score))

    sample_scores.sort(key=lambda x: x[1], reverse=True)
    permuted_indices = [idx for idx, _ in sample_scores]

    rearranged = [raw_json[i] for i in permuted_indices]
    allocations = refit(rearranged)

    with open(CONFIG["output_json"], "w") as f:
        json.dump(rearranged, f, indent=4)

    print(f"Predictions saved to {CONFIG['output_json']}")

def refit(allocs):
    stations = len(allocs[0]["data"])
    overlaps = 0
    timeline = {s: [] for s in range(stations)}
    allocations = []

    for seq_num, obj in enumerate(allocs):
        start_x = seq_num * takt
        for station_key, size in obj["data"].items():
            station = int(station_key[1:]) - 1
            offset = olov_offset(seq_num, station, size, allocations)
            offset = obj["offsets"][station_key]
            obj.setdefault("offsets", {})[station_key] = offset
            x_start = start_x + offset
            x_end = x_start + size

            alloc = Allocation(0, seq_num, seq_num + station, station, size, offset)

            for prev_start, prev_end in timeline[station]:
                if x_start < prev_end and x_end > prev_start:
                    overlaps += 1

            timeline[station].append((x_start, x_end))
            allocations.append(alloc)

    print(f"Total overlaps detected: {overlaps}")
    return allocations

def olov_offset(seq_num, station, size, allocations):
    n_l, n_r = find_neighbours_left(seq_num, station, allocations)
    return int(max(n_l, n_r))

def find_neighbours_left(seq_num, station, allocations):
    neighbours_left, neighbours_right = -50, -50
    for x in allocations:
        seq, stn = x.get_coords()
        s, o = x.get_data()
        if seq == seq_num and stn == station - 1:
            edge_l = int(-takt + s + o)
            neighbours_left = edge_l
        if seq == seq_num - 1 and stn == station:
            edge_r = int(-takt + s + o)
            neighbours_right = edge_r
    return neighbours_left, neighbours_right

if __name__ == "__main__":
    main()
