import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from ML.Transformer.model import build_transformer
from ML.Transformer.dataset import Dataset 
from ML.Transformer.config import get_config
import random

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # Set to CPU for initial portability


def load_model(config):
    dataset = Dataset(config["training_path"], train_frac = 1.0)

    src0, tgt0 = dataset[0]

    print(src0)
    print(tgt0)

    num_items = src0.shape[0]
    input_dim = src0.shape[1]

    print("\nobjects: \t", len(dataset))
    print("num_items: \t", num_items)
    print("input_dim: \t", input_dim, "\n")
    vocab_size = num_items * input_dim

    model = build_transformer(
            src_vocab_size = 1, 
            tgt_vocab_size = 1, 
            src_seq_len = num_items, 
            tgt_seq_len = num_items, 
            d_model = config["d_model"]
            ).to(device)

    return model, dataset

class ObjectEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(4, d_model)
    
    def forward(self, x):
        # x: [batch, stations, 4] -> [batch, stations, d_model]
        return self.linear(x)

def custom_loss(output_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    # output_scores shape: [batch_size, seq_len, 1] or [batch_size]
    # target_scores shape: [batch_size]
    
    if output_scores.dim() == 3:
        predicted_scores = output_scores.mean(dim=1).squeeze(-1) 
    elif output_scores.dim() == 1:
        # [batch]
        predicted_scores = output_scores
    else:
        predicted_scores = output_scores.mean(dim = list(range(1, output_scores.dim()))).squeeze(-1)

    loss = F.mse_loss(predicted_scores, target_scores.float(), reduction='mean')

    return loss


def Train():
    model, dataset = load_model(config)
    object_embedder = ObjectEmbedding(config["d_model"]).to(device)

    model.src_embed = object_embedder # object_embedder returns [batch, stations, d_model]
    model.tgt_embed = nn.Identity() # tgt is already embedded, no need to re-embed.
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=dataset.collate_fn)
    train_data = dataset.train_data
    train_targets = dataset.train_targets
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        
        combined_train = list(zip(train_data, train_targets))
        
        for batch_idx, (batch_objects, batch_targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            batch_objects = batch_objects.to(device) # [batch, stations, 4]
            batch_targets = batch_targets.to(device) # [batch] - the single object ID


            src = batch_objects # [batch, stations, 4]
            tgt = src # [batch, stations, 4]
            
            encoder_output = model.encode(src, src_mask=None) # [batch, seq_len, d_model]
            
            tgt_embed_out = model.src_embed(tgt) # [batch, stations, d_model]
            tgt_embed_out = model.src_pos(tgt_embed_out) 
            
            decoder_output = model.decode(encoder_output, src_mask=None, tgt=tgt_embed_out, tgt_mask=None) 
            # decoder_output: [batch, seq_len, d_model]
            
            output = model.project(decoder_output)  # [batch, seq_len, vocab_size=1]
            
            loss = custom_loss(output, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch+1} Loss: {avg_loss:.4f}")
    
    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_folder / f"epoch_{epoch:02d}.pt")
    
    print(f"Model saved to {model_folder / f'epoch_{epoch:02d}.pt'}")

Train()
