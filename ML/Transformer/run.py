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
from ML.Transformer.model import build_transformer
from ML.Transformer.dataset import Dataset
from ML.Transformer.config import get_config

alloc_config = alloc_con()
config = get_config()

DATASET_SCALING = 1000.0
DRIFT_LIMIT = float(alloc_config["drift"]) 

class ObjectEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.station_embed = nn.Embedding(1000, d_model // 2)
        self.size_embed = nn.Linear(1, d_model // 2)
        self.proj_in_dim = (d_model // 2) * 2 
        self.out_proj = nn.Linear(self.proj_in_dim, d_model)
    
    def forward(self, x):
        s_idx = x[:, :, 0].long()
        s_val = x[:, :, 1].unsqueeze(-1) * 10.0
        emb_idx = self.station_embed(s_idx)
        emb_val = self.size_embed(s_val)
        cat = torch.cat([emb_idx, emb_val], dim=-1)
        return self.out_proj(cat)

def load_model_for_inference(device, dataset, epoch_to_load="best_model"):
    src0, _ = dataset[0]
    num_items = src0.shape[0]

    model = build_transformer(
        src_vocab_size=1,
        tgt_vocab_size=1,
        src_seq_len=num_items,
        tgt_seq_len=num_items,
        d_model=config["d_model"]
    ).to(device)

    model.src_embed = ObjectEmbedding(config["d_model"]).to(device)
    
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
    sample_tensor = sample_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        enc_out = model.encode(sample_tensor, src_mask=None)
        logits = model.project(enc_out)
        drift_values = logits * DATASET_SCALING
        return drift_values.squeeze(0).cpu().numpy().flatten()

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
                # --- FIX IS HERE ---
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
    model = load_model_for_inference(device, dataset, epoch_to_load="best_model")

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
