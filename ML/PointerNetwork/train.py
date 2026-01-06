import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    
    dataset = PointerDataset(config["training_path"], epoch_multiplier=200)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=dataset.collate_fn)
    
    model = PointerNetwork(
        input_dim=config["input_dim"], 
        hidden_dim=config["hidden_dim"],
        d_model=config.get("d_model", 256)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_loss = float('inf')

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for inputs, targets in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            logits, _ = model(inputs, targets)
            # Flatten
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(config["model_folder"]) / "best_ptr.pt")

if __name__ == "__main__":
    train()
