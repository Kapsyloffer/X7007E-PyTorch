import json
import torch
from torch.utils.data import Dataset as TorchDataset
import random
from torch.nn.utils.rnn import pad_sequence
from scripts.config import get_obj_config

config = get_obj_config()

class PointerDataset(TorchDataset):
    def __init__(self, json_path_or_data, train_frac=0.9, shuffle=True, max_seq_len=100):
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
                keys = [f"s{i}" for i in range(1, config["stations"]+1)] 
                obj_features = [obj["data"][k] / 1000.0 for k in keys]
                sequence_tensor.append(obj_features)

            if len(sequence_tensor) > 1:
                for i in range(0, len(sequence_tensor), max_seq_len):
                    chunk = sequence_tensor[i:i + max_seq_len]
                    if len(chunk) < 2:
                        continue
                    self.samples.append(torch.tensor(chunk, dtype=torch.float))
                    self.targets.append(torch.tensor(list(range(len(chunk))), dtype=torch.long))

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
        # Sortera efter längd (krävs ofta för pack_padded_sequence om du använder det senare)
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        samples = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Padda samples med 0.0 till den längsta sekvensen i batchen
        padded_samples = pad_sequence(samples, batch_first=True, padding_value=0.0)
        
        # Padda targets med -1 (eller annat index som CrossEntropyLoss ignorerar via ignore_index)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
        
        return padded_samples, padded_targets
