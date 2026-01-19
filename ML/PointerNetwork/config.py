from pathlib import Path

def get_config():
    return {
        "training_path": "jsons/allocations.json",
        "run_path": "jsons/shuffled.json",
        "output_json": "jsons/predicted.json",
        "batch_size": 128,
        "num_epochs": 10,
        "lr": 1e-4,
        
        # Model Architecture
        "input_dim": 35,      # 35 Station Sizes (TODO: LINK TO OTHER CONFIG)
        "hidden_dim": 512,    # LSTM Memory size
        "d_model": 256,       # Embedding size 
        
        "model_folder": "ML/weights_ptr",
    }
