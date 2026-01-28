import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
     sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Updated imports for Transformer
from ML.Transformer.model import build_transformer
from ML.Transformer.dataset import PointerDataset
from ML.Transformer.config import get_config

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_masks(src, tgt_indices):
    # src: (B, S, F)
    # src_mask: (B, 1, 1, S)
    src_mask = (src.abs().sum(dim=-1) > 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
    
    # decoder mask
    seq_len = tgt_indices.size(1)
    nopeak_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.bool)
    tgt_mask = nopeak_mask.to(device) == 0
    
    return src_mask.to(device), tgt_mask

def Train():
    dataset = PointerDataset(config["training_path"], max_seq_len=100)
    batch_size = min(config["batch_size"], len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    model = build_transformer(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        N=config["n_layers"],
        h=config["n_heads"],
        dropout=config["dropout"],
        d_ff=config["d_ff"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(config["num_epochs"]):
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_samples, batch_targets in iterator:
            batch_samples = batch_samples.to(device)
            batch_targets = batch_targets.to(device)

            src_mask, tgt_mask = make_masks(batch_samples, batch_targets)

            # Teacher Forcing
            logits = model(batch_samples, tgt_indices=batch_targets, src_mask=src_mask, tgt_mask=tgt_mask)
            
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch_targets.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    torch.save(model.state_dict(), model_folder / "best_model.pt")
    print(f"Model saved to {model_folder / 'best_model.pt'}")

if __name__ == "__main__":
    Train()
