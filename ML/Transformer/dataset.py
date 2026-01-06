import json
import torch
from torch.utils.data import Dataset as TorchDataset
import random

class Dataset(TorchDataset):
    def __init__(self, json_path_or_data, train_frac=0.9, shuffle=True, augment=False):
        if isinstance(json_path_or_data, str):
            with open(json_path_or_data, "r") as f:
                raw = json.load(f)
        else:
            raw = json_path_or_data
        
        self.augment = augment
        self.samples = [] 
        self.targets = []

        for entry in raw:
            keys = sorted(entry["data"].keys(), key=lambda x: int(x[1:]))
            
            sample_tensor = []
            target_tensor = []

            for k in keys:
                station_idx = int(k[1:])
                size_val = entry["data"][k] / 1000.0
                
                if "offsets" in entry:
                    offset_val = entry["offsets"][k] / 1000.0
                else:
                    offset_val = 0.0

                sample_tensor.append([station_idx, size_val])
                target_tensor.append([offset_val])

            self.samples.append(torch.tensor(sample_tensor, dtype=torch.float))
            self.targets.append(torch.tensor(target_tensor, dtype=torch.float))

        # Train / val split
        total_samples = len(self.samples)
        train_size = int(train_frac * total_samples)

        if shuffle:
            combined = list(zip(self.samples, self.targets))
            random.shuffle(combined)
            self.samples, self.targets = zip(*combined)

        self.train_data = self.samples[:train_size]
        self.train_targets = self.targets[:train_size]
        self.val_data = self.samples[train_size:]
        self.val_targets = self.targets[train_size:]

        print(f"Loaded {len(self.samples)} samples. Augment={self.augment}")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample = self.train_data[idx].clone()
        target = self.train_targets[idx].clone()

        if self.augment:
             # 1. Jitter: Add small noise to sizes
             noise = (torch.rand_like(sample[:, 1]) * 0.02) - 0.01 
             sample[:, 1] += noise

             # 2. Station Dropout: 30% chance to zero out one station's size
             if random.random() < 0.3:
                 mask_idx = random.randint(0, sample.shape[0]-1)
                 sample[mask_idx, 1] = 0.0

        return sample, target

    def get_val_data(self):
        return self.val_data, self.val_targets

    def collate_fn(self, batch):
        samples = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)
        return samples, targets
