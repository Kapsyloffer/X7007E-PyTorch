import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from model import build_transformer
from dataset import Dataset 
from config import get_config
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

    # The Transformer architecture inherently supports permutation generation
    # if trained with appropriate masking and a loss that encourages a sequence output.
    # However, given the current setup of predicting a single label ID in the original script,
    # we'll restructure the output for an arbitrary ranking/scoring task.
    
    # We will use num_items (e.g., 50 stations) as vocab_size if the goal is to predict 
    # the correct sequence of stations. However, since the current task seems to be 
    # ranking the input objects (1 to 100), we'll aim for a scalar output representing a score.
    # The current build_transformer uses vocab_size for the final dimension of the projection layer.
    
    # We temporarily set src/tgt vocab size to 1 to force the output to be a scalar/single-value projection.
    model = build_transformer(
            src_vocab_size = 1, # Dummy size, replaced by NumericInputWrapper later
            tgt_vocab_size = 1, # Dummy size, replaced by NumericInputWrapper later
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
    
    # Custom loss function placeholder.
    # For a ranking objective, a loss should encourage the scores of 'better' 
    # permutations (or, in this simplified case, inputs) to be higher.
    # Example: MSELoss might be used if target_scores were continuous quality metrics.
    # Example: Listwise ranking losses (like ListNet, ListMLE) if processing multiple samples.
    # Since we are currently training against a single object ID which doesn't
    # represent a score, we must design the actual objective.
    #
    # *** IMPORTANT ***
    # Since the original target is a sequential ID (1, 2, 3, ...), 
    # a simple score prediction task can be formulated by having the 
    # model predict a simple regression target that we invent, e.g., the object ID itself,
    # and use MSE loss. This is a very rough proxy for a ranking objective.
    #
    # For now, this loss is set to return a placeholder result, preventing runtime errors.
    # It must be replaced with your actual loss function for permutation/ranking.

    #########################################################################
    # START CUSTOM LOSS BLOCK
    # TODO: Implement your custom permutation loss here.
    # Placeholder: Simple Mean Squared Error, assuming the goal is to predict
    # a continuous score that should correlate with the original target ID.
    
    # Reshape prediction to match target dimension for MSE: [batch_size, seq_len] -> [batch_size]
    # Taking the mean across the sequence length dim (dim=1) to get one score per batch item.
    
    # Note: If the actual goal is permutation learning, this approach is highly limited.
    # A true permutation loss would typically compare the ranks of different permutations.
    
    # For a placeholder, let's aim for a single scalar output per sample as predicted score.
    
    if output_scores.dim() == 3:
         # [batch, seq_len, 1] -> [batch]
        predicted_scores = output_scores.mean(dim=1).squeeze(-1) 
    elif output_scores.dim() == 1:
        # [batch]
        predicted_scores = output_scores
    else:
        # Fallback to an un-ideal mean, adjust based on your actual model output structure.
        predicted_scores = output_scores.mean(dim = list(range(1, output_scores.dim()))).squeeze(-1)

    loss = F.mse_loss(predicted_scores, target_scores.float(), reduction='mean')

    # END CUSTOM LOSS BLOCK
    #########################################################################
    
    return loss


def Train():
    model, dataset = load_model(config)
    object_embedder = ObjectEmbedding(config["d_model"]).to(device)

    # Replace the token-based embeddings with the numerical input wrapper.
    # We must ensure that the inner Linear layer in InputEmbeddings/NumericInputWrapper
    # is still trained, but since we modify it here, we use Identity().
    model.src_embed = object_embedder # object_embedder returns [batch, stations, d_model]
    model.tgt_embed = nn.Identity() # tgt is already embedded, no need to re-embed.
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=dataset.collate_fn)
    # Note: Validation loader uses the original dataset, may not be correct if you need to load from a different file.
    # dataset.get_val_data() returns (val_data, val_targets) but DataLoader expects a Dataset object.
    # Reverting to iterating over full dataset and manually splitting for simplicity here.
    # If the provided `get_val_data` works with DataLoader in your environment, use it instead.
    
    # For simplicity (since dataset class doesn't seem to be iterable directly for validation)
    train_data = dataset.train_data
    train_targets = dataset.train_targets
    
    # Optimizer for both the core transformer parameters and the linear layer inside the object embedder
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    # Since the original target is the object ID (an index), we need a continuous 
    # target for the current permutation-ranking-by-score approach. We use the original 
    # target ID as the pseudo-score for simplicity in this example. 
    # The loss function above implements this using MSELoss as a placeholder.

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        
        # We need to manually combine the train_data and train_targets for iteration
        combined_train = list(zip(train_data, train_targets))
        # Add a placeholder for permutation logic later
        
        for batch_idx, (batch_objects, batch_targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            batch_objects = batch_objects.to(device) # [batch, stations, 4]
            batch_targets = batch_targets.to(device) # [batch] - the single object ID

            # Permutation Logic for Training (Example)
            # You would implement your permutation/augmentation logic here.
            # Example: Shuffle the stations for 'src' but keep 'tgt' original for an auto-regressive task.
            # For this simple regression-style scoring: No explicit permutation needed, 
            # as the model learns to map the whole input tensor to a score.

            src = batch_objects # [batch, stations, 4]
            # Since the model uses the output of src_embed, and src_embed is ObjectEmbedding (a Linear layer)
            # the shape becomes [batch, stations, d_model]
            
            # For decoder input, since the output vocab size is artificially 1,
            # we need a dummy target sequence for the decoder. Let's use the encoder's input tensor.
            # This is a common simplification in auto-encoders or non-standard transformer applications
            # where a sequence output is required but the sequence itself is not autoregressive.
            tgt = src # [batch, stations, 4]
            
            encoder_output = model.encode(src, src_mask=None) # [batch, seq_len, d_model]
            
            # The tgt for decode must have been processed by tgt_embed
            # If tgt is set to src, tgt_embed (Identity) returns [batch, stations, 4], 
            # but the decoder is expecting [batch, seq_len, d_model] (after embedding) 
            # Let's adjust by encoding tgt the same way as src to maintain dimensional consistency.
            # This turns the whole system into a massive auto-encoder/feature extractor.
            
            # Re-initialize tgt on the fly with the ObjectEmbedding layer before passing to decode.
            # Since ObjectEmbedding is already set to model.src_embed:
            # Note: We skip tgt_pos(tgt) to maintain simplicity given the numerical nature
            # and the current sequential ID target.
            
            # The model architecture demands that the target input sequence be embedded 
            # *before* being passed to the decoder.
            # We enforce this by manually running `model.src_embed` (which is the object_embedder)
            # over the target data (which is the same as source data) as a simplification.
            tgt_embed_out = model.src_embed(tgt) # [batch, stations, d_model]
            tgt_embed_out = model.src_pos(tgt_embed_out) # Re-use positional encoding for decoder input.
            
            decoder_output = model.decode(encoder_output, src_mask=None, tgt=tgt_embed_out, tgt_mask=None) 
            # decoder_output: [batch, seq_len, d_model]
            
            output = model.project(decoder_output)  # [batch, seq_len, vocab_size=1]
            
            # Calculate the custom loss
            # output shape: [batch, stations, 1]
            # batch_targets shape: [batch]
            loss = custom_loss(output, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch+1} Loss: {avg_loss:.4f}")
    
    # Save the model state dictionary
    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_folder / f"epoch_{epoch:02d}.pt")
    
    print(f"Model saved to {model_folder / f'epoch_{epoch:02d}.pt'}")

# We wrap the existing call in an if block for standard Python module usage.
if __name__ == "__main__":
    Train()
