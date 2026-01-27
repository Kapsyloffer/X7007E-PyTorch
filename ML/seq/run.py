import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import copy
from pathlib import Path
from tqdm import tqdm
import sys

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import get_config as alloc_con
from model import build_seq2seq
from dataset import Dataset
from config import get_config

alloc_config = alloc_con()
config = get_config()

DATASET_SCALING = 1000.0
DRIFT_LIMIT = float(alloc_config["drift"]) 

# Re-defining Wrapper for Inference consistency
class LSTMWrapper(nn.Module):
    def __init__(self, seq2seq_model, d_model):
        super().__init__()
        self.model = seq2seq_model
        self.station_embed = nn.Embedding(1000, d_model // 2)
        self.size_embed = nn.Linear(1, d_model // 2)
        self.input_proj = nn.Linear((d_model // 2) * 2, d_model)
        self.target_proj = nn.Linear(1, 1) 
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.0):
        # src: (Batch, Seq, 2)
        s_idx = src[:, :, 0].long()
        s_val = src[:, :, 1].unsqueeze(-1) * 10.0
        
        emb_idx = self.station_embed(s_idx)
        emb_val = self.size_embed(s_val)
        cat = torch.cat([emb_idx, emb_val], dim=-1)
        src_emb = self.input_proj(cat) 
        
        return self.model(src_emb, tgt, teacher_forcing_ratio)
    
    def encode(self, src):
        s_idx = src[:, :, 0].long()
        s_val = src[:, :, 1].unsqueeze(-1) * 10.0
        emb_idx = self.station_embed(s_idx)
        emb_val = self.size_embed(s_val)
        cat = torch.cat([emb_idx, emb_val], dim=-1)
        src_emb = self.input_proj(cat)
        return self.model.encoder(src_emb)

def load_model_for_inference(device, epoch_to_load="best_model"):
    d_model = config["d_model"]
    
    core_model = build_seq2seq(
        input_dim=d_model, 
        hidden_dim=d_model, 
        output_dim=1, 
        num_layers=2, 
        dropout=0.0,
        device=device
    )
    
    model = LSTMWrapper(core_model, d_model).to(device)
    
    if "best" in epoch_to_load:
        model_name = "best_model.pt"
    else:
        model_name = f"epoch_{epoch_to_load}.pt"
        
    model_path = Path(config["model_folder"]) / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model

def predict_offsets(model, sample_tensor, device):
    model.eval()
    with torch.no_grad():
        sample_tensor = sample_tensor.unsqueeze(0).to(device) # (1, Seq, 2)
        
        # 1. Encode
        encoder_outputs, hidden, cell = model.encode(sample_tensor)
        
        # 2. Decode Step-by-Step
        decoder_input = torch.zeros(1, 1, 1).to(device) # SOS
        
        outputs = []
        max_len = sample_tensor.shape[1]
        
        for _ in range(max_len):
            output, hidden, cell, _ = model.model.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(output.item())
            
            # Autoregressive input for next step
            decoder_input = output
            
        drift_values = np.array(outputs) * DATASET_SCALING
        return drift_values

def calculate_conflict(offsets_a, offsets_b):
    combined = np.abs(offsets_a) + np.abs(offsets_b)
    violations = np.maximum(0, combined - DRIFT_LIMIT)
    return np.sum(violations ** 2)

def calculate_total_score(sequence, cost_matrix):
    score = 0.0
    for k in range(len(sequence) - 1):
        i = sequence[k]
        j = sequence[k+1]
        score += cost_matrix[i, j]
    return score

def simulated_annealing(ids, cost_matrix):
    current_seq = copy.deepcopy(ids)
    random.shuffle(current_seq)
    
    current_score = calculate_total_score(current_seq, cost_matrix)
    best_seq = list(current_seq)
    best_score = current_score
    
    T_start = 1000.0
    T_end = 0.001
    alpha = 0.995 
    max_steps = 50000 
    
    T = T_start
    
    pbar = tqdm(range(max_steps), desc="Annealing")
    
    for step in pbar:
        idx1, idx2 = random.sample(range(len(current_seq)), 2)
        
        new_seq = list(current_seq)
        new_seq[idx1], new_seq[idx2] = new_seq[idx2], new_seq[idx1]
        
        new_score = calculate_total_score(new_seq, cost_matrix)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_seq = new_seq
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_seq = list(current_seq) 
        
        T *= alpha
        if T < T_end:
            break
            
        if step % 1000 == 0:
            pbar.set_postfix({"Best Cost": f"{best_score:.2f}", "Temp": f"{T:.2f}"})
            
    return best_seq

def run_transformer(run_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = run_path if run_path is not None else config["run_path"]
    
    with open(path, "r") as f:
        processing_json = json.load(f)

    dataset = Dataset(processing_json, train_frac=1.0, shuffle=False, augment=False)
    model = load_model_for_inference(device, epoch_to_load="best_model")

    num_objs = len(dataset)
    all_offsets = []
    
    for idx in range(num_objs):
        sample_tensor, _ = dataset[idx] 
        offsets = predict_offsets(model, sample_tensor, device)
        all_offsets.append(offsets)
        
    cost_matrix = np.zeros((num_objs, num_objs))
    
    for i in tqdm(range(num_objs), desc="Building Matrix"):
        for j in range(num_objs):
            if i == j: 
                cost_matrix[i, j] = float('inf') 
            else:
                cost_matrix[i, j] = calculate_conflict(all_offsets[i], all_offsets[j])

    initial_indices = list(range(num_objs))
    optimized_indices = simulated_annealing(initial_indices, cost_matrix)
    
    with open(path, "r") as f:
        clean_json = json.load(f)

    rearranged = [clean_json[i] for i in optimized_indices]

    if run_path is not None and run_path != config["run_path"]:
        return rearranged
    
    output_path = config.get("output_json", "predicted.json")
    with open(output_path, "w") as f:
        json.dump(rearranged, f, indent=4)

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    run_transformer(None)
