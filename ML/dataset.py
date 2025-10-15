import json
import torch
from torch.utils.data import Dataset

class ItemReorderingDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self.samples = []
        self.targets = []

        for entry in raw:
            data = entry["data"]
            offsets = entry.get("offsets", {k: 0 for k in data.keys()})

            # Input tensor: [[size, offset], ...]
            x = [[data[k], offsets.get(k, 0)] for k in sorted(data.keys())]
            x = torch.tensor(x, dtype=torch.float)

            # Target permutation: original order
            perm = torch.arange(len(x), dtype=torch.long)

            self.samples.append(x)
            self.targets.append(perm)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
