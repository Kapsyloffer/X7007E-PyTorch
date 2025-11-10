import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import json
import argparse

from model import build_transformer
from dataset import ItemReorderingDataset


class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha  

    def overlap_penalty(self, predictions, stations, obj_spacing):
        penalty = 0.0
        placed_rects = {y: [] for y in range(stations)}  

        for obj_idx in range(predictions.size(0)):  
            for station_idx in range(stations):
                pred_start = predictions[obj_idx, station_idx, 0].item()
                pred_end = pred_start + predictions[obj_idx, station_idx, 1].item()  

                for prev_start, prev_end in placed_rects[station_idx]:
                    if pred_start < prev_end and pred_end > prev_start:
                        penalty += 10  

                placed_rects[station_idx].append((pred_start, pred_end))

        return penalty

    def forward(self, logits, tgt, predictions, stations, obj_spacing):
        ce_loss = self.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        overlap_loss = self.overlap_penalty(predictions, stations, obj_spacing)
        total_loss = ce_loss + self.alpha * overlap_loss
        return total_loss


class NumericInputWrapper(nn.Module):
    #(size, offset) -> d_model
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.proj(x)


def get_config():
    return {
        "json_path": "jsons/allocations.json",
        "batch_size": 8,
        "num_epochs": 20,
        "d_model": 256,
        "lr": 1e-5,
        "model_folder": "weights",
        "experiment_name": "runs/reordering_transformer"
    }


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ItemReorderingDataset(config["json_path"], train_frac=0.8)

    train_dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset.get_val_data(), batch_size=config["batch_size"], shuffle=False)

    num_stations = dataset[0][0].shape[0]  # Stations
    input_dim = dataset[0][0].shape[1]  # size + offset; 2.
    output_dim = num_stations

    # Build transformer with original model
    model = build_transformer(
        src_vocab_size=num_stations,  # placeholder
        tgt_vocab_size=output_dim,
        src_seq_len=num_stations,
        tgt_seq_len=num_stations,
        d_model=config["d_model"]
    ).to(device)

    # Replace embeddings with numeric projections
    model.src_embed = NumericInputWrapper(input_dim=input_dim, d_model=config["d_model"]).to(device)
    model.tgt_embed = NumericInputWrapper(input_dim=input_dim, d_model=config["d_model"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = CustomLoss(alpha=10.0)  # Use the custom loss function with an overlap penalty

    Path(config["model_folder"]).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(config["experiment_name"])
    global_step = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        # Train loop
        for src, tgt in tqdm(train_dataloader, desc=f"Epoch {epoch:02d}"):
            src, tgt = src.to(device), tgt.to(device)

            # Create src_mask and tgt_mask (set to None if not used)
            src_mask = None
            tgt_mask = None

            enc_out = model.encode(src.float(), src_mask)  # Now passing the masks
            dec_out = model.decode(enc_out, src_mask, src.float(), tgt_mask)  # Now passing the masks
            logits = model.project(dec_out)  # (B, seq_len, seq_len)

            # Assuming that the model output is of shape (B, seq_len, 2) where it predicts (x_start, x_end)
            predictions = logits  # Adjust if necessary to match your output format

            # Compute loss with overlap penalty
            loss = loss_fn(logits, tgt, predictions, stations=num_stations, obj_spacing=700)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_dataloader:
                src, tgt = src.to(device), tgt.to(device)

                # Create src_mask and tgt_mask (set to None if not used)
                src_mask = None
                tgt_mask = None

                # Forward pass
                enc_out = model.encode(src.float(), src_mask)
                dec_out = model.decode(enc_out, src_mask, src.float(), tgt_mask)
                logits = model.project(dec_out)

                # Compute loss
                val_loss += loss_fn(logits, tgt, logits, stations=num_stations, obj_spacing=700).item()

        val_loss /= len(val_dataloader)
        writer.add_scalar("val_loss", val_loss, global_step)

        # Save model per epoch
        torch.save(model.state_dict(), Path(config["model_folder"]) / f"epoch_{epoch:02d}.pt")

    writer.close()


if __name__ == "__main__":
    config = get_config()
    train_model(config)
