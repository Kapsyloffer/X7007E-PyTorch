import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from model import build_transformer
from dataset import Dataset

from config import get_config


class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, takt=700):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.takt = takt

    def overlap_penalty(self, pred, sizes):
        # pred: (B, T)
        B, T = pred.shape
        penalty = 0.0
        for b in range(B):
            counts = torch.bincount(pred[b], minlength=T)
            dup = torch.clamp(counts - 1, min=0)  # >0 for duplicates
            penalty += dup.sum()
        return penalty / B

    def forward(self, logits, tgt, sizes):
        B, T, V = logits.shape
        logits = logits[:, :T, :]
        ce = self.ce(logits.reshape(B * T, V), tgt.reshape(B * T))
        pred = logits.argmax(dim=-1)
        ov = self.overlap_penalty(pred, sizes)
        return ce + self.alpha * ov


class NumericInputWrapper(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.proj(x)




def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset(config["json_path"], train_frac=0.8)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset.get_val_data(), batch_size=config["batch_size"], shuffle=False)

    src0, tgt0 = dataset[0]

    num_items = src0.shape[0]
    input_dim = src0.shape[1]

    print("num_items: ", num_items)
    print("input_dim: ", input_dim)

    model = build_transformer(
        src_vocab_size=num_items,
        tgt_vocab_size=num_items,
        src_seq_len=num_items,
        tgt_seq_len=num_items,
        d_model=config["d_model"]
    ).to(device)

    model.src_embed = NumericInputWrapper(input_dim, config["d_model"]).to(device)
    model.tgt_embed = NumericInputWrapper(input_dim, config["d_model"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = CustomLoss(alpha=config["alpha"])

    Path(config["model_folder"]).mkdir(exist_ok=True, parents=True)
    global_step = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            src = src.float().to(device)
            tgt = tgt.long().to(device)

            sizes = src[..., 0]

            enc = model.encode(src, None)
            dec = model.decode(enc, None, src, None)

            logits = model.project(dec)
            loss = loss_fn(logits, tgt, sizes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.float().to(device)
                tgt = tgt.long().to(device)

                sizes = src[..., 0]

                enc = model.encode(src, None)
                dec = model.decode(enc, None, src, None)

                logits = model.project(dec)
                val_loss += float(loss_fn(logits, tgt, sizes))

        val_loss /= len(val_loader)

        torch.save(model.state_dict(), Path(config["model_folder"]) / f"epoch_{epoch:02d}.pt")



if __name__ == "__main__":
    config = get_config()
    train_model(config)
