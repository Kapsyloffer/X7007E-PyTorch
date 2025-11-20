import json
import torch
from torch.utils.data import Dataset, random_split

class ItemReorderingDataset(Dataset):
    def __init__(self, json_path, train_frac=0.8):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self.samples = []
        self.targets = []

        for entry in raw:
            data = entry["data"]
            offsets = entry.get("offsets", {k: 0 for k in data.keys()})

            x = [[data[k], offsets.get(k, 0)] for k in sorted(data.keys())]
            x = torch.tensor(x, dtype=torch.float)

            # Target permutation: original order
            perm = torch.arange(len(x), dtype=torch.long)

            self.samples.append(x)
            self.targets.append(perm)

        # Create a dataset split for training and validation
        total_samples = len(self.samples)
        train_size = int(train_frac * total_samples)
        val_size = total_samples - train_size

        # Randomly split the dataset into training and validation sets
        self.train_data = list(zip(self.samples[:train_size], self.targets[:train_size]))
        self.val_data = list(zip(self.samples[train_size:], self.targets[train_size:]))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def get_val_data(self):
        return self.val_data
