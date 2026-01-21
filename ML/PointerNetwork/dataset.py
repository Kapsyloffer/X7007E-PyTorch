import json
import torch
from torch.utils.data import Dataset as TorchDataset
import random

class PointerDataset(TorchDataset):
    def __init__(self, json_path_or_data, train_frac=0.9, shuffle=True):
        if isinstance(json_path_or_data, str):
            with open(json_path_or_data, "r") as f:
                raw = json.load(f)
        else:
            raw = json_path_or_data
        
        self.samples = []
        self.targets = []

        # Packa in platt JSON i en lista om det behövs
        sequences = [raw] if isinstance(raw, list) and isinstance(raw[0], dict) else raw

        for sequence in sequences:
            if not isinstance(sequence, list): continue
            
            sequence_tensor = []
            for obj in sequence:
                # Extrahera stationer s1..s35 i strikt ordning
                keys = [f"s{i}" for i in range(1, 36)]
                obj_features = [obj["data"][k] / 1000.0 for k in keys]
                sequence_tensor.append(obj_features)

            if len(sequence_tensor) > 0:
                self.samples.append(torch.tensor(sequence_tensor, dtype=torch.float))
                self.targets.append(torch.tensor(list(range(len(sequence_tensor))), dtype=torch.long))

        total = len(self.samples)
        train_size = max(1, int(train_frac * total))

        if shuffle and total > 1:
            combined = list(zip(self.samples, self.targets))
            random.shuffle(combined)
            self.samples, self.targets = zip(*combined)

        self.train_data = self.samples[:train_size]
        self.train_targets = self.targets[:train_size]
        self.val_data = self.samples[train_size:]
        self.val_targets = self.targets[train_size:]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample = self.train_data[idx].clone()
        
        # För att lära modellen sekvensering shufflar vi indata
        indices = list(range(sample.size(0)))
        random.shuffle(indices)
        
        shuffled_sample = sample[indices]
        # Target blir de index som pekar ut rätt objekt i den shufflade listan
        new_target = torch.tensor([indices.index(i) for i in range(len(indices))], dtype=torch.long)
        
        return shuffled_sample, new_target

    def get_val_data(self):
        return self.val_data, self.val_targets

    def collate_fn(self, batch):
        samples = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)
        return samples, targets
