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
    dataset = Dataset(config["training_path"], train_frac = 1.0)

    src0, tgt0 = dataset[0]

    print(src0)
    print(tgt0)

    num_items = src0.shape[0]
    input_dim = src0.shape[1]

    print("\nobjects: \t", len(dataset))
    print("num_items: \t", num_items)
    print("input_dim: \t", input_dim, "\n")

    # Create mapping from unique [size, offset] pairs to integer IDs for embedding
    unique_pairs = set()
    for sample in dataset.samples:
        for pair in sample.tolist():
            unique_pairs.add(tuple(pair))
    global pair2id

    pair2id = {pair: i for i, pair in enumerate(unique_pairs)}
    vocab_size = len(pair2id)

    print("Vocab size (unique [size, offset] pairs):", vocab_size)

    model = build_transformer(
            src_vocab_size = vocab_size,
            tgt_vocab_size = vocab_size,
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
        # x: [batch, stations, 2]
        return self.linear(x)

def Train():
    model, dataset = load_model(config)
    object_embedder = ObjectEmbedding(config["d_model"]).to(device)

    model.src_embed = nn.Identity()
    model.tgt_embed = nn.Identity()
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(dataset.get_val_data(), batch_size=config["batch_size"], shuffle=False, collate_fn=dataset.collate_fn)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(object_embedder.parameters()), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        for batch_objects, batch_targets in train_loader:
            batch_objects = batch_objects.to(device)
            batch_targets = batch_targets.to(device)

            batch_embed = object_embedder(batch_objects)
            
            src = batch_embed
            tgt = batch_embed
            
            encoder_output = model.encode(src, src_mask=None)
            decoder_output = model.decode(encoder_output, src_mask=None, tgt=tgt, tgt_mask=None)
            output = model.project(decoder_output)  # [batch, seq_len, vocab_size]
            
            # flatten for loss: (batch*seq_len, vocab_size) vs (batch*seq_len)
            batch_targets_seq = batch_targets.unsqueeze(1).repeat(1, output.size(1))
            loss = criterion(output.view(-1, output.size(-1)), batch_targets_seq.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print("Epoch:", epoch, "Loss:", total_loss)
    torch.save(model.state_dict(), Path(config["model_folder"]) / f"epoch_{epoch:02d}.pt")
Train()
