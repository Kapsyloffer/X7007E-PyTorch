import json
import torch
from torch.utils.data import Dataset as TorchDataset
import random

class Dataset(TorchDataset):
    def __init__(self, json_path, train_frac=0.8):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self.samples = []  # [obj_id, station_idx, data, offset]
        self.targets = []

        last = raw[0]["object"]
        stack = None
        x = None

        for entry in raw:
            keys = sorted(entry["data"].keys())

            if(entry["object"] is last):
                y = torch.tensor([[entry["object"], i+1, entry["data"][k], entry["offsets"][k]] for i, k in enumerate(keys)], dtype = torch.float)
                if(x is not None):
                    x = torch.cat((x, y), dim=0)
                else:
                    x = y
            else:
                self.samples.append(x)
                last = entry["object"]
                x = torch.tensor([[entry["object"], i+1, entry["data"][k], entry["offsets"][k]] for i, k in enumerate(keys)], dtype = torch.float)
        self.samples.append(x)

        self.targetize(self.samples)

        # print(self.samples)
        # Train / val split
        total_samples = len(self.samples)
        train_size = int(train_frac * total_samples)

        self.train_data = self.samples[:train_size]
        self.train_targets = self.targets[:train_size]

        self.val_data = self.samples[train_size:]
        self.val_targets = self.targets[train_size:]


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # Return tuple of (data, target)
        return self.train_data[idx], torch.tensor(self.train_targets[idx], dtype=torch.long)

    def get_val_data(self):
        return self.val_data, self.val_targets

    def targetize(self, tensors):
        samples = []
        targets = []
        i = 1
        for x in tensors:
            samples.append(x)
            targets.append(i)
            i = i+1
        
        combined = list(zip(samples, targets))
        random.shuffle(combined)
        samples, targets = zip(*combined)
        samples = list(samples)
        targets = list(targets)

        # print("\nsamples: ", samples, "\n targets: ", targets)

        self.samples = samples 
        self.targets = targets

    def collate_fn(self, batch):
        samples = torch.stack([item[0] for item in batch], dim=0)  # [batch, stations, 4]
        targets = torch.stack([item[1] for item in batch], dim=0)  # [batch]
        return samples, targets
