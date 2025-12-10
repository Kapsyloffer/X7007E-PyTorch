import json
import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, json_path, train_frac=0.8):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self.samples = []  # [obj_id, station_idx, data, offset]

        last = raw[0]["object"]
        stack = None
        x = None

        for entry in raw:
            keys = sorted(entry["data"].keys())

            if(entry["object"] is last or last is None):
                y = torch.tensor([[entry["object"], i, entry["data"][k], entry["offsets"][k]] for i, k in enumerate(keys)], dtype = torch.float)
                if(x is not None):
                    x = torch.cat((x, y), dim=0)
                else:
                    x = y
            else:
                print(x)
                
                self.samples.append(x)

                last = entry["object"]
                x = torch.tensor([[entry["object"], i, entry["data"][k], entry["offsets"][k]] for i, k in enumerate(keys)], dtype = torch.float)
        print(self.samples)
        # Train / val split
        total_samples = len(self.samples)
        train_size = int(train_frac * total_samples)

        self.train_data = self.samples[:train_size]
        self.val_data = self.samples[train_size:]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def get_val_data(self):
        return self.val_data
    
