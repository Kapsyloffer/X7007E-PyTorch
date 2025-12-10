import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from model import build_transformer
from dataset import Dataset
from config import get_config

config = get_config()

class NumericInputWrapper(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.proj(x)


def load_model(device, dataset):
    num_stations = dataset[0][0].shape[0]
    input_dim = dataset[0][0].shape[1]

    model = build_transformer(
        src_vocab_size=num_stations,
        tgt_vocab_size=num_stations,
        src_seq_len=num_stations,
        tgt_seq_len=num_stations,
        d_model=config["d_model"]
    ).to(device)

    model.src_embed = NumericInputWrapper(input_dim=input_dim, d_model=config["d_model"]).to(device)
    model.tgt_embed = NumericInputWrapper(input_dim=input_dim, d_model=config["d_model"]).to(device)

    return model


def predict_score(model, sample_tensor, device):
    sample_tensor = sample_tensor.unsqueeze(0).float().to(device)
    with torch.no_grad():
        enc_out = model.encode(sample_tensor, src_mask=None)
        dec_out = model.decode(enc_out, src_mask=None, tgt=sample_tensor, tgt_mask=None)
        logits = model.project(dec_out)
        return logits.mean().item()


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    with open(config["run_path"], "r") as f:
        raw_json = json.load(f)

    dataset = Dataset(config["run_path"])
    model = load_model(device, dataset)
    model.eval()

    scores = []
    for idx, (sample_tensor, _) in tqdm(enumerate(dataset), total=len(dataset), desc="Scoring samples"):
        scores.append((idx, predict_score(model, sample_tensor, device)))

    scores.sort(key=lambda x: x[1], reverse=True)
    permuted_indices = [idx for idx, _ in scores]

    rearranged = [raw_json[i] for i in permuted_indices]

    with open(config["output_json"], "w") as f:
        json.dump(rearranged, f, indent=4)

    print(f"Predictions saved to {config['output_json']}")


if __name__ == "__main__":
    main()
