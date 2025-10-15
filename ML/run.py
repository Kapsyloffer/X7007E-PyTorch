import torch
from pathlib import Path
import json
from model import build_transformer
from dataset import ItemReorderingDataset
import torch.nn as nn
from tqdm import tqdm

# Numeric input wrapper (same as in training)
class NumericInputWrapper(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.proj(x)

CONFIG = {
    "json_path": "jsons/allocations.json",
    "model_folder": "weights",
    "d_model": 256,
    "epoch": 9,  # checkpoint epoch
    "output_json": "jsons/predicted.json"
}

def load_model(device, dataset):
    num_stations = dataset[0][0].shape[0]
    input_dim = dataset[0][0].shape[1]

    model = build_transformer(
        src_vocab_size=num_stations,
        tgt_vocab_size=num_stations,
        src_seq_len=num_stations,
        tgt_seq_len=num_stations,
        d_model=CONFIG["d_model"]
    ).to(device)

    # Replace embeddings with numeric projection
    model.src_embed = NumericInputWrapper(input_dim=input_dim, d_model=CONFIG["d_model"]).to(device)
    model.tgt_embed = NumericInputWrapper(input_dim=input_dim, d_model=CONFIG["d_model"]).to(device)

    model_path = Path(CONFIG["model_folder"]) / f"epoch_{CONFIG['epoch']:02d}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def predict_sample_scores(model, sample_tensor, device):
    """
    Returns a single score per sample by averaging over sequence logits.
    sample_tensor: (1, seq_len, features)
    """
    sample_tensor = sample_tensor.unsqueeze(0).to(device)  # add batch dim
    tgt_tensor = sample_tensor  # placeholder

    with torch.no_grad():
        enc_out = model.encode(sample_tensor.float(), src_mask=None)
        dec_out = model.decode(enc_out, src_mask=None, tgt=tgt_tensor.float(), tgt_mask=None)
        logits = model.project(dec_out)  # (1, seq_len, seq_len)
        score = logits.mean().item()  # scalar score
    return score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load JSON
    with open(CONFIG["json_path"], "r") as f:
        raw_json = json.load(f)

    dataset = ItemReorderingDataset(CONFIG["json_path"])
    model = load_model(device, dataset)

    # Predict a score for each sample (we can use this to rank/reorder)
    sample_scores = []
    for idx in tqdm(range(len(dataset)), desc="Scoring samples"):
        sample_tensor, _ = dataset[idx]
        score = predict_sample_scores(model, sample_tensor, device)
        sample_scores.append((idx, score))

    # Sort by score descending to produce permutation
    sample_scores.sort(key=lambda x: x[1], reverse=True)
    permuted_indices = [idx for idx, _ in sample_scores]

    # Reorder raw JSON
    rearranged = [raw_json[i] for i in permuted_indices]

    # Save
    with open(CONFIG["output_json"], "w") as f:
        json.dump(rearranged, f, indent=4)

    print(f"Predictions saved to {CONFIG['output_json']}")

if __name__ == "__main__":
    main()
