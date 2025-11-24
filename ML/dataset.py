import json
import torch
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

class Dataset(Dataset):
    def __init__(self, json_path, train_frac=0.8):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self.samples = [] # Tensor av objekten: [stations, 2] där 2 är [data, offset]
        self.targets = [] # OG ordning


        for i, entry in enumerate(raw):
            keys = sorted(entry["data"].keys())

            x = torch.tensor(
                [[entry["data"][k], entry["offsets"][k]] for k in keys], dtype = torch.float
            ) 

            self.samples.append(x)
            self.targets.append(i)
        
        self.targets = torch.tensor(self.targets, dtype=torch.long)


        # Create a dataset split for training and validation
        total_samples = len(self.samples)
        indices = torch.randperm(total_samples) # <--- Skeptisk till den här TODO: Undersök med och utan den.

        train_size = int(train_frac * total_samples)
        train_idx = indices[:train_size]

        val_size = indices[train_size:]

        # Randomly split the dataset into training and validation sets
        self.train_data = list(zip(self.samples[:train_size], self.targets[:train_size]))
        self.val_data = list(zip(self.samples[train_size:], self.targets[train_size:]))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def get_val_data(self):
        return self.val_data
