from pathlib import Path

def get_config():
    return {
        "json_path": "jsons/allocations.json",
        "output_json": "jsons/predicted.json",
        "batch_size": 8,
        "num_epochs": 10,
        "d_model": 256,
        "lr": 1e-5,
        "alpha": 0.1,
        "model_folder": "ML/weights",
    }
