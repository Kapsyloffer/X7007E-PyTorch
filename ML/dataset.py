import json
import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, json_path, train_frac=0.8):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self.samples = []  # [stations, 2]  (2 = [data, offset])
        self.targets = []  # Tensor of permutation indices for sequence ordering

        for entry in raw:
            keys = sorted(entry["data"].keys())
            x = torch.tensor(
                [[entry["data"][k], entry["offsets"][k]] for k in keys], dtype = torch.float
            )
            self.samples.append(x)

            # initial target = identity permutation (same length as num_stations)
            self.targets.append(torch.arange(len(x), dtype=torch.long))

        # Train / val split
        total_samples = len(self.samples)
        train_size = int(train_frac * total_samples)

        self.train_data = list(zip(self.samples[:train_size], self.targets[:train_size]))
        self.val_data = list(zip(self.samples[train_size:], self.targets[train_size:]))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def get_val_data(self):
        return self.val_data
    
    def collate_fn(self, batch):
        samples = torch.stack([item[0] for item in batch], dim=0)  # [batch, stations, 2]
        targets = torch.stack([item[1] for item in batch], dim=0)  # [batch, stations]
        return samples, targets

