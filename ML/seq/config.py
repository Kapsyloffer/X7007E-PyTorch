from pathlib import Path

def get_config():
    return {
        "training_path": "jsons/allocations.json",
        "run_path": "jsons/shuffled.json",
        "output_json": "jsons/predicted.json",
        "batch_size": 16,
        "num_epochs": 10,
        "lr": 1e-3,

        "d_model": 64, 
        "dropout": 0.1, 
        "model_folder": "ML/weights_seq",
    }

