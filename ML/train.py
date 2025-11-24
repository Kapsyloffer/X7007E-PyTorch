import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_transformer
from dataset import Dataset 

from config import get_config

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

def load_model(config):
    dataset = Dataset(config["json_path"], train_frac = 0.8)

    src0, tgt0 = dataset[0]

    # for i in range(len(dataset)):
    #     print("dataset #",i, "\n", dataset[i], "\n")
    
    num_items = src0.shape[0]
    input_dim = src0.shape[1]

    print("objects: \t", len(dataset))
    print("num_items: \t", num_items)
    print("input_dim: \t", input_dim)

    # Create mapping from unique [size, offset] pairs to integer IDs for embedding
    unique_pairs = set()
    for sample in dataset.samples:
        for pair in sample.tolist():  # sample = Tensor([num_stations, 2])
            unique_pairs.add(tuple(pair))  # make hashable
    global pair2id
    pair2id = {pair: i for i, pair in enumerate(unique_pairs)}
    vocab_size = len(pair2id)
    print("Vocab size (unique [size, offset] pairs):", vocab_size)

    model = build_transformer(
            src_vocab_size = vocab_size,  # use mapped vocab size
            tgt_vocab_size = vocab_size,
            src_seq_len = num_items, 
            tgt_seq_len = num_items, 
            d_model = config["d_model"]
            ).to(device)

    return model, dataset

# Optional: keep ObjectEmbedding in case you want numeric linear projection instead
class ObjectEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2, d_model)
    
    def forward(self, x):
        # x: [num_objects, num_stations, 2] => 2 = [size, offset]
        # output: (num_objects, num_stations, d_model)
        return self.linear(x)

# Convert batch of numeric [size, offset] pairs to integer IDs
def batch_to_indices(batch_objects):
    # batch_objects: [batch_size, num_stations, 2] float tensor
    # returns: [batch_size, num_stations] LongTensor
    batch_indices = []
    for sample in batch_objects:           # sample: [num_stations, 2]
        idxs = [pair2id[tuple(pair.tolist())] for pair in sample]
        batch_indices.append(idxs)
    return torch.tensor(batch_indices, dtype=torch.long) if batch_indices else torch.empty(0, dtype=torch.long)

def Train():
    model, dataset = load_model(config)
    object_embedder = ObjectEmbedding(config["d_model"]).to(device)

# Patch: bypass embedding layers
    model.src_embed = nn.Identity()
    model.tgt_embed = nn.Identity()
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset.get_val_data(), batch_size=config["batch_size"], shuffle=False)
    
    # model.parameters() = all trainable weights of transformer.
    # object_embedder.parameters() = all weights of the embedding linear layer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(object_embedder.parameters()), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        # training loop
        # batch_objects Tensor([num_stations, 2]) where 2 = [size, offset]
        for batch_objects, batch_targets in train_loader:
            batch_objects = batch_objects.to(device)   # numeric input
            batch_targets = batch_targets.to(device)
            
            # [batch_size, num_stations, d_model]
            batch_embed = object_embedder(batch_objects)
            
            # [batch_size, 1, d_model]; one vector per object as "sequence length 1"
            src = batch_embed.mean(dim=1, keepdim=True)
            tgt = batch_embed.mean(dim=1, keepdim=True)
            
            # Forward pass: numeric vectors go directly to encoder/decoder
            encoder_output = model.encode(src, src_mask=None)
            decoder_output = model.decode(encoder_output, src_mask=None, tgt=tgt, tgt_mask=None)
            output = model.project(decoder_output).squeeze(1)  # [batch_size, num_classes]
            
            loss = criterion(output, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print("Epoch:", epoch, "Loss:", total_loss)

Train()

