from pathlib import Path

def get_config():
    return {
        "training_path": "jsons/allocations.json",
        "run_path": "jsons/shuffled.json",
        "output_json": "jsons/predicted.json",
        "batch_size": 8,
        "num_epochs": 50,
        "d_model": 256,
        "lr": 1e-6,
        "alpha": 0.2,
        "model_folder": "ML/weights",
    }
