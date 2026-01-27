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

from ML.seq.model import build_seq2seq
from ML.seq.dataset import SeqDataset
from ML.seq.config import get_config

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Train():
    dataset = SeqDataset(config["training_path"], max_seq_len=100)
    batch_size = min(config["batch_size"], len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # output_dim set to 100 to handle sorting indices for sequences up to 100 length
    model = build_seq2seq(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=100, 
        num_layers=2,
        dropout=config["dropout"],
        device=device
    )

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

            # Updated call with correct kwarg 'tgt'
            logits, _ = model(batch_samples, tgt=batch_targets)
            
            # Reshape logits to (Batch * SeqLen, NumClasses)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
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
