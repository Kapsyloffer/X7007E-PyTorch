import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from model import build_transformer
from dataset import ItemReorderingDataset

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
        "num_epochs": 10,
        "d_model": 256,
        "lr": 1e-4,
        "model_folder": "weights",
        "experiment_name": "runs/reordering_transformer"
    }

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ItemReorderingDataset(config["json_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    num_stations = dataset[0][0].shape[0]
    input_dim = dataset[0][0].shape[1]  # size + offset
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
    loss_fn = nn.CrossEntropyLoss()

    Path(config["model_folder"]).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(config["experiment_name"])
    global_step = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        for src, tgt in tqdm(dataloader, desc=f"Epoch {epoch:02d}"):
            src, tgt = src.to(device), tgt.to(device)

            # masks not needed for now
            src_mask = None
            tgt_mask = None

            enc_out = model.encode(src.float(), src_mask)
            dec_out = model.decode(enc_out, src_mask, src.float(), tgt_mask)
            logits = model.project(dec_out)  # (B, seq_len, seq_len)

            loss = loss_fn(logits.view(-1, output_dim), tgt.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # Save model per epoch
        torch.save(model.state_dict(), Path(config["model_folder"]) / f"epoch_{epoch:02d}.pt")

    writer.close()

if __name__ == "__main__":
    config = get_config()
    train_model(config)

