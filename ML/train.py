import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from model import build_transformer
from dataset import ItemReorderingDataset


class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, takt=700):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.takt = takt

    def overlap_penalty(self, pred_perm, sizes):
        B, N = pred_perm.shape
        penalty = 0.0
        stations = torch.arange(N).unsqueeze(0).repeat(B, 1)

        for b in range(B):
            perm = pred_perm[b].cpu().numpy()
            sz = sizes[b].cpu().numpy()
            stn = stations[b].cpu().numpy()
            allocations = []
            timeline = {s: [] for s in range(max(stn)+1)}

            for seq_num, idx in enumerate(perm):
                start_x = seq_num * self.takt
                station = stn[idx]

                offset = 0
                n_l, n_r = -50, -50
                for alloc in allocations:
                    prev_seq, prev_stn, s, o = alloc
                    if prev_seq == seq_num and prev_stn == station - 1:
                        n_l = max(n_l, -self.takt + s + o)
                    if prev_seq == seq_num - 1 and prev_stn == station:
                        n_r = max(n_r, -self.takt + s + o)
                offset = max(n_l, n_r, 0)

                x_start = start_x + offset
                x_end = x_start + sz[idx]

                for ps, pe in timeline[station]:
                    if x_start < pe and x_end > ps:
                        penalty += 1.0

                timeline[station].append((x_start, x_end))
                allocations.append((seq_num, station, sz[idx], offset))

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


def get_config():
    return {
        "json_path": "jsons/allocations.json",
        "batch_size": 4,
        "num_epochs": 10,
        "d_model": 256,
        "lr": 1e-5,
        "alpha": 0.1,
        "model_folder": "weights",
        "experiment_name": "runs/reordering_transformer"
    }


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ItemReorderingDataset(config["json_path"], train_frac=0.8)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset.get_val_data(), batch_size=config["batch_size"], shuffle=False)

    src0, tgt0 = dataset[0]

    num_items = src0.shape[0]
    input_dim = src0.shape[1]

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
    writer = SummaryWriter(config["experiment_name"])
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

            writer.add_scalar("train_loss", float(loss.detach()), global_step)
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
        writer.add_scalar("val_loss", val_loss, global_step)

        torch.save(model.state_dict(), Path(config["model_folder"]) / f"epoch_{epoch:02d}.pt")

    writer.close()


if __name__ == "__main__":
    config = get_config()
    train_model(config)
