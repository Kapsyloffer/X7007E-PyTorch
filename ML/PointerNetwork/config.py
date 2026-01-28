from pathlib import Path
from scripts.config import get_obj_config

obj_config = get_obj_config()

def get_config():
    return {
        "training_path": "jsons/allocations.json",
        "run_path": "jsons/shuffled.json",
        "output_json": "jsons/predicted.json",
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 1e-4,
        
        # Model Architecture
        "input_dim": obj_config["stations"],
        "hidden_dim": 128,    # LSTM Memory size
        "d_model": 256,       # Embedding size 
        
        "model_folder": "ML/weights_ptr",
    }
