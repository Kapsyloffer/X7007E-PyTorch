from pathlib import Path
from scripts.config import get_obj_config

obj_config = get_obj_config()

def get_config():
    return {
        "training_path": "jsons/allocations.json",
        "run_path": "jsons/shuffled.json",
        "output_json": "jsons/predicted.json",
        "batch_size": 16,
        "num_epochs": 10,
        "lr": 1e-3,
        
        # Model Architecture
        "input_dim": obj_config["stations"],      # 35 Station Sizes (TODO: LINK TO OTHER CONFIG)
        "d_model": 256,       # Embedding size 
        "dropout": 0.1,
        "d_ff": 512,          # Feed Forward expansion dimension
        "n_heads": 8,         # Number of attention heads
        "n_layers": 3,        # Number of Encoder/Decoder blocks
        
        "model_folder": "ML/weights_trans", 

    }
