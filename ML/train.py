import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path

from model import build_transformer
from dataset import Dataset 
from config import get_config

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

def load_model(config):
    dataset = Dataset(config["training_path"], train_frac = 0.8)

    src0, tgt0 = dataset[0]

    num_items = src0.shape[0]
    input_dim = src0.shape[1]

    print("objects: \t", len(dataset))
    print("num_items: \t", num_items)
    print("input_dim: \t", input_dim)


    model = build_transformer(
        src_vocab_size = num_items,
        tgt_vocab_size = num_items,
        src_seq_len = num_items, 
        tgt_seq_len = num_items, 
        d_model = config["d_model"]
        ).to(device)

    return model, dataset

class ObjectEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2, d_model)
    
    def forward(self, x):
        # x: [batch, stations, 2]
        return self.linear(x)


def dup_penalty(pred):
    # pred: [B, T]
    B, T = pred.shape
    penalty = 0.0
    for b in range(B):
        c = torch.bincount(pred[b], minlength=T)
        d = torch.clamp(c - 1, min=0)
        penalty += d.sum()
    return penalty / B


def Train():
    model, dataset = load_model(config)
    object_embedder = ObjectEmbedding(config["d_model"]).to(device)

    model.src_embed = nn.Identity()
    model.tgt_embed = nn.Identity()
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(dataset.get_val_data(), batch_size=config["batch_size"], shuffle=False, collate_fn=dataset.collate_fn)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(object_embedder.parameters()), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        for batch_objects, batch_targets in train_loader:
            batch_objects = batch_objects.to(device)
            batch_targets = batch_targets.to(device)

            batch_embed = object_embedder(batch_objects)
            
            src = batch_embed
            tgt = batch_embed

            T = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(device)
            
            encoder_output = model.encode(src, src_mask=None)
            decoder_output = model.decode(encoder_output, src_mask=None, tgt=tgt, tgt_mask=tgt_mask)
            output = model.project(decoder_output)  # [batch, seq_len, vocab_size]
            

            ce_loss = loss_fn(output.view(-1, output.size(-1)), batch_targets.view(-1))
            pred = output.argmax(-1)
            ov = dup_penalty(pred)

            print(pred)

            loss = ce_loss + config["alpha"] * ov

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
        
        print("Epoch:", epoch, "Loss:", total_loss)

    torch.save(model.state_dict(), Path(config["model_folder"]) / f"epoch_{epoch:02d}.pt")

Train()
