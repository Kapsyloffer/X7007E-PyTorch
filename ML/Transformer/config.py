from pathlib import Path

def get_config():
    return {
        "training_path": "jsons/allocations.json",
        "run_path": "jsons/shuffled.json",
        "output_json": "jsons/predicted.json",
        "batch_size": 8,
        "num_epochs": 100,
        "d_model": 512, 
        "lr": 1e-6,
        "dropout": 0.3, 
        "model_folder": "ML/weights",
    }
