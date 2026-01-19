import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import sys

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model import PointerNetwork
from dataset import PointerDataset
from config import get_config

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

def train():
    
    # Gradient Accumulation Steps
    ACCUM_STEPS = 16
    batch_size = max(1, config["batch_size"] // ACCUM_STEPS)

    dataset = PointerDataset(config["training_path"], epoch_multiplier=200)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    
    model = PointerNetwork(
        input_dim=config["input_dim"], 
        hidden_dim=config["hidden_dim"],
        d_model=config.get("d_model", 256)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scaler = GradScaler('cuda')
    
    best_loss = float('inf')

    torch.cuda.empty_cache()

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for i, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward
            with autocast('cuda'):
                logits, _ = model(inputs, targets)
                # Flatten
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss / ACCUM_STEPS
            
            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUM_STEPS
            iterator.set_postfix({"loss": f"{loss.item() * ACCUM_STEPS:.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(config["model_folder"]) / "best_ptr.pt")

if __name__ == "__main__":
    train()
