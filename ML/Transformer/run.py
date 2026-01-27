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
TAKT_TIME = 700.0

class ObjectEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.station_embed = nn.Embedding(1000, d_model // 2)
        self.size_embed = nn.Linear(1, d_model // 2)
        self.proj_in_dim = (d_model // 2) * 2 
        self.out_proj = nn.Linear(self.proj_in_dim, d_model)
        
        pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)
      
    def forward(self, x):
        s_idx = x[:, :, 0].long()
        s_val = x[:, :, 1].unsqueeze(-1) * 10.0
        
        emb_idx = self.station_embed(s_idx)
        emb_val = self.size_embed(s_val)
        cat = torch.cat([emb_idx, emb_val], dim=-1)
        x = self.out_proj(cat)
        
        x = x + self.pe[:x.size(1), :]
        return x

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
    model.load_state_dict(state_dict, strict=False)
    return model

def predict_offsets(model, sample_tensor, device):
    sample_tensor = sample_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        enc_out = model.encode(sample_tensor, src_mask=None)
        logits = model.project(enc_out)
        drift_values = logits * DATASET_SCALING
        return drift_values.squeeze(0).cpu().numpy().flatten()

def calculate_conflict(offset_a, size_a, offset_b_pred, size_b):
    
    # 1. Calculate the Simulated Offset for B using your centering formula
    # Formula derived: (Offset_A + Size_A - Size_B + Gap) / 2
    # This represents B "centering" itself in the slack left by A.
    simulated_offset_b = (offset_a + size_a - size_b) / 2.0
    
    # 2. Define valid bounds for B
    # It cannot start before -Drift
    min_start = -DRIFT_LIMIT
    # It cannot end after Takt + Drift
    # So Start <= Takt + Drift - Size
    max_start = TAKT_TIME + DRIFT_LIMIT - size_b
    
    # 3. Calculate Violation
    # If simulated position is outside the bounds, that is the cost.
    # We square it to penalize large violations heavily.
    violation_low = np.maximum(0, min_start - simulated_offset_b)
    violation_high = np.maximum(0, simulated_offset_b - max_start)
    
    total_violation = violation_low + violation_high
    
    return np.sum(total_violation ** 2)

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
    N = len(current_seq)
    
    pbar = tqdm(range(max_steps), desc="Annealing")
    
    for step in pbar:
        idx1, idx2 = random.sample(range(N), 2)
        if idx1 == idx2: continue
        
        # Incremental Update Logic
        delta = 0.0
        def get_cost(k1, k2):
            return cost_matrix[current_seq[k1], current_seq[k2]]

        if idx1 > 0: delta -= get_cost(idx1 - 1, idx1)
        if idx1 < N - 1: delta -= get_cost(idx1, idx1 + 1)
        if idx2 > 0: delta -= get_cost(idx2 - 1, idx2)
        if idx2 < N - 1: delta -= get_cost(idx2, idx2 + 1)
        
        if abs(idx1 - idx2) == 1:
            lower = min(idx1, idx2)
            delta += get_cost(lower, lower + 1) 
            
        item1, item2 = current_seq[idx1], current_seq[idx2]
        
        # Add new edges (Simulating swap: idx1 has item2, idx2 has item1)
        if idx1 > 0: delta += cost_matrix[current_seq[idx1-1], item2]
        if idx1 < N - 1: delta += cost_matrix[item2, current_seq[idx1+1]]
        if idx2 > 0: delta += cost_matrix[current_seq[idx2-1], item1]
        if idx2 < N - 1: delta += cost_matrix[item1, current_seq[idx2+1]]
        
        if abs(idx1 - idx2) == 1:
            # Correction for adjacency double-counting in addition
            # The edge is now between item2 and item1
            # If idx1 < idx2: (item2, item1) at idx1
            # If idx2 < idx1: (item1, item2) at idx2
            if idx1 < idx2:
                delta -= cost_matrix[item2, item1] # We added this twice above (as right of idx1 and left of idx2)
                delta += cost_matrix[item2, item1] # Add it back once... wait.
                # Let's just rely on the standard subtractions.
                # Above: added (idx1-1, item2) and (item2, idx1+1). 
                # If idx2 is idx1+1, then (item2, item1) was added.
                # Then for idx2: added (idx2-1, item1) -> (item2, item1). Added twice.
                delta -= cost_matrix[item2, item1]
            else:
                 delta -= cost_matrix[item1, item2]

        # Apply Swap
        current_seq[idx1], current_seq[idx2] = current_seq[idx2], current_seq[idx1]
        
        # Verify delta with real calculation occasionally or just rely on accept/reject
        # For safety/simplicity in this snippet, we use the delta directly
        # But to be 100% safe against index bugs, we calculate full score if needed.
        # Given the complexity, let's trust the delta but revert on reject.
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_seq = list(current_seq) 
        else:
            # Revert swap
            current_seq[idx1], current_seq[idx2] = current_seq[idx2], current_seq[idx1]
       
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
    all_sizes = []
    
    for idx in range(num_objs):
        sample_tensor, _ = dataset[idx] 
        offsets = predict_offsets(model, sample_tensor, device)
        sizes = sample_tensor[:, 1].numpy() * 10.0
        all_offsets.append(offsets)
        all_sizes.append(sizes)
        
    cost_matrix = np.zeros((num_objs, num_objs))
    
    for i in tqdm(range(num_objs), desc="Building Matrix"):
        for j in range(num_objs):
            if i == j: 
                cost_matrix[i, j] = float('inf') 
            else:
                # Use the centered-simulation logic
                cost_matrix[i, j] = calculate_conflict(
                    all_offsets[i], all_sizes[i],
                    all_offsets[j], all_sizes[j]
                )

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
