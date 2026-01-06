import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset as TorchDataset

class PointerDataset(TorchDataset):
    def __init__(self, json_path, epoch_multiplier=100):
        with open(json_path, "r") as f:
            self.raw_data = json.load(f)
            
        self.multiplier = epoch_multiplier
        self.full_features = []
        
        for entry in self.raw_data:
            keys = sorted(entry["data"].keys(), key=lambda x: int(x[1:]))
            feats = [entry["data"][k] / 1000.0 for k in keys] 
            self.full_features.append(feats)
            
        self.num_items = len(self.full_features)
            
    def __len__(self):
        return self.multiplier * 1

    def __getitem__(self, idx):
        # Curriculum: Train on random sub-sequence lengths (20 to Max)
        # This prevents the model from overfitting to a fixed length
        current_len = random.randint(20, self.num_items)
        
        # 1. Pick random subset
        subset_indices = sorted(random.sample(range(self.num_items), current_len))
        subset_features = [self.full_features[i] for i in subset_indices]
        
        # 2. Shuffle the subset to create the puzzle
        shuffled_local_indices = np.random.permutation(current_len)
        shuffled_input = [subset_features[i] for i in shuffled_local_indices]
        
        # 3. Solve it 
        target_pointers = np.argsort(shuffled_local_indices)
        
        return torch.tensor(shuffled_input, dtype=torch.float), torch.tensor(target_pointers, dtype=torch.long)

    def collate_fn(self, batch):
        inputs = [x[0] for x in batch]
        targets = [x[1] for x in batch]
        
        inputs_pad = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets_pad = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)
        
        return inputs_pad, targets_pad
