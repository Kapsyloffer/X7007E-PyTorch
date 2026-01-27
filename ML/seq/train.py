import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import math

from model import build_seq2seq
from dataset import Dataset
from config import get_config

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_SCALING = 1000.0
DRIFT_LIMIT = 200

# Input dim = 2 (id, size), but we project it in the training loop
# Actually, let's keep it simple and use embedding layers inside the train loop or model
# For LSTM, we usually just pass the features directly or embed them.

class LSTMWrapper(nn.Module):
    def __init__(self, seq2seq_model, d_model):
        super().__init__()
        self.model = seq2seq_model
        # We need to project our raw input (ID, Size) to d_model size for the LSTM
        self.station_embed = nn.Embedding(1000, d_model // 2)
        self.size_embed = nn.Linear(1, d_model // 2)
        self.input_proj = nn.Linear((d_model // 2) * 2, d_model)
        
        # We also need to project the target (1 value) to d_model for the Decoder Input
        self.target_proj = nn.Linear(1, 1) # Keeping it 1-to-1 for decoder input size
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (Batch, Seq, 2)
        s_idx = src[:, :, 0].long()
        s_val = src[:, :, 1].unsqueeze(-1) * 10.0
        
        emb_idx = self.station_embed(s_idx)
        emb_val = self.size_embed(s_val)
        cat = torch.cat([emb_idx, emb_val], dim=-1)
        src_emb = self.input_proj(cat) # (Batch, Seq, d_model)
        
        return self.model(src_emb, tgt, teacher_forcing_ratio)

def combined_loss(predictions: torch.Tensor, targets: torch.Tensor, alpha_factor: float) -> torch.Tensor:
    l1_loss = F.l1_loss(predictions, targets, reduction='mean')
    
    current_item = predictions[:, :-1, 0] 
    next_item = predictions[:, 1:, 0]
    
    combined_drift = (torch.abs(current_item) + torch.abs(next_item)) * DATASET_SCALING
    violations = F.relu(combined_drift - DRIFT_LIMIT)
    conflict_loss = torch.mean((violations / DATASET_SCALING) ** 2) 
    
    return l1_loss + (alpha_factor * conflict_loss)

def validate(wrapper, val_loader, alpha_factor):
    wrapper.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_src, batch_tgt in val_loader:
            src = batch_src.to(device)
            tgt = batch_tgt.to(device) 
            
            # 0.0 teacher forcing ratio for validation (fully autoregressive)
            output = wrapper(src, tgt, teacher_forcing_ratio=0.0)
            
            val_loss = combined_loss(output, tgt, alpha_factor)
            total_val_loss += val_loss.item()
            
    wrapper.train()
    return total_val_loss / len(val_loader)

def Train():
    dataset = Dataset(config["training_path"], train_frac=0.9, shuffle=False, augment=False)
    
    # Dimensions
    d_model = config["d_model"]
    
    # Build core Seq2Seq
    # Input to encoder is d_model (after embedding)
    # Output of decoder is 1 (the offset)
    core_model = build_seq2seq(
        input_dim=d_model, 
        hidden_dim=d_model, 
        output_dim=1, 
        num_layers=2, 
        dropout=config.get("dropout", 0.1),
        device=device
    )
    
    model = LSTMWrapper(core_model, d_model).to(device)
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=dataset.collate_fn)
    
    val_data_raw, val_targets_raw = dataset.get_val_data()
    val_dataset = torch.utils.data.TensorDataset(torch.stack(val_data_raw), torch.stack(val_targets_raw))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    model.train() 
    best_val_loss = float('inf')
    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["num_epochs"]):
        total_loss = 0
        iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        progress = epoch / config["num_epochs"]
        alpha_factor = min(1.0, progress * 2.0)
        
        # Linear decay of teacher forcing from 0.5 to 0.0
        tf_ratio = max(0.0, 0.5 - (0.5 * progress))

        for batch_idx, (batch_objects, batch_targets) in iterator:
            batch_objects = batch_objects.to(device) 
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            
            output = model(batch_objects, batch_targets, teacher_forcing_ratio=tf_ratio)
            
            loss = combined_loss(output, batch_targets, alpha_factor)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        current_val_loss = validate(model, val_loader, alpha_factor)
        
        scheduler.step(current_val_loss)
        
        print(f"Epoch: {epoch+1} | Train Loss: {avg_loss:.6f} | Val Loss: {current_val_loss:.6f}")

        if current_val_loss < best_val_loss:
             best_val_loss = current_val_loss
             torch.save(model.state_dict(), model_folder / "best_model.pt")
    
    print(f"Best Model saved to {model_folder / 'best_model.pt'}")

if __name__ == "__main__":
    Train()
