import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import math

# if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
#     sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import get_config as alloc_con
from ML.Transformer.model import build_transformer
from ML.Transformer.dataset import Dataset 
from ML.Transformer.config import get_config

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

alloc_config = alloc_con()
config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_SCALING = 1000.0
DRIFT_LIMIT = float(alloc_config["drift"]) / DATASET_SCALING 

def load_model(config):
    dataset = Dataset(config["training_path"], train_frac=0.9, shuffle=True, augment=False)
    src0, _ = dataset[0]
    num_items = src0.shape[0] 

    model = build_transformer(
            src_vocab_size = 1, 
            tgt_vocab_size = 1, 
            src_seq_len = num_items, 
            tgt_seq_len = num_items, 
            d_model = config["d_model"],
            dropout = config.get("dropout", 0.1)
            ).to(device)

    return model, dataset

class ObjectEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Standard embedding size (no mask token needed)
        self.station_embed = nn.Embedding(1000, d_model // 2)
        self.size_embed = nn.Linear(1, d_model // 2)
        self.proj_in_dim = (d_model // 2) * 2 
        self.out_proj = nn.Linear(self.proj_in_dim, d_model)
    
    def forward(self, x):
        s_idx = x[:, :, 0].long()
        s_val = x[:, :, 1].unsqueeze(-1) * 10.0 
        
        emb_idx = self.station_embed(s_idx)
        emb_val = self.size_embed(s_val)
        cat = torch.cat([emb_idx, emb_val], dim=-1)
        return self.out_proj(cat)

def combined_loss(predictions: torch.Tensor, targets: torch.Tensor, alpha_factor: float) -> torch.Tensor:
    l1_loss = F.l1_loss(predictions, targets, reduction='mean')
    
    current_item = predictions[:, :-1, 0] 
    next_item = predictions[:, 1:, 0]     
    combined_drift = torch.abs(current_item) + torch.abs(next_item)
    
    violations = F.relu(combined_drift - DRIFT_LIMIT)
    conflict_loss = torch.mean(violations ** 2) 
    
    return l1_loss + (alpha_factor * conflict_loss)

def validate(model, val_data, val_targets, alpha_factor):
    model.eval()
    with torch.no_grad():
        src = val_data.to(device)
        tgt = val_targets.to(device) 
        
        # Encoder-Only Validation
        enc_out = model.encode(src, src_mask=None)
        output = model.project(enc_out) 
        val_loss = combined_loss(output, tgt, alpha_factor)
    model.train()
    return val_loss.item()

def Train():
    model, dataset = load_model(config)
    model.src_embed = ObjectEmbedding(config["d_model"]).to(device)
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=dataset.collate_fn)
    
    val_data_raw, val_targets_raw = dataset.get_val_data()
    val_data = torch.stack(val_data_raw)
    val_targets = torch.stack(val_targets_raw)

    accumulation_steps = 4 

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)
    
    actual_steps_per_epoch = len(train_loader) // accumulation_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4, 
        epochs=config["num_epochs"],
        steps_per_epoch=actual_steps_per_epoch
    )
    
    model.train() 
    best_val_loss = float('inf')
    # patience = math.floor(config["num_epochs"] /6)
    trigger_times = 0

    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["num_epochs"]):
        total_loss = 0
        iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        progress = epoch / config["num_epochs"]
        alpha_factor = min(1.0, progress * 2.0)

        for batch_idx, (batch_objects, batch_targets) in iterator:
            batch_objects = batch_objects.to(device) 
            batch_targets = batch_targets.to(device)

            encoder_output = model.encode(batch_objects, src_mask=None)
            output = model.project(encoder_output)
            
            loss = combined_loss(output, batch_targets, alpha_factor)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            iterator.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        avg_loss = total_loss / len(train_loader)
        current_val_loss = validate(model, val_data, val_targets, alpha_factor)
        
        print(f"Epoch: {epoch+1} | Train Loss: {avg_loss:.6f} | Val Loss: {current_val_loss:.6f}")

        # if current_val_loss < best_val_loss:
        #     best_val_loss = current_val_loss
        #     trigger_times = 0
        #     torch.save(model.state_dict(), model_folder / "best_model.pt")
        # else:
        #     trigger_times += 1
        #     print(trigger_times, "/", patience)
        #     if trigger_times >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break
    
    print(f"Best Model saved to {model_folder / 'best_model.pt'}")

if __name__ == "__main__":
    Train()
