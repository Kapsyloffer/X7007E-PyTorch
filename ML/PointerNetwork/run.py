import json
import torch
from pathlib import Path
from tqdm import tqdm
import sys

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model import PointerNetwork
from config import get_config

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pointer():
    input_path = config["run_path"]
    
    with open(input_path, "r") as f:
        raw_data = json.load(f)
        
    input_features = []
    for entry in raw_data:
        keys = sorted(entry["data"].keys(), key=lambda x: int(x[1:]))
        feats = [entry["data"][k] / 1000.0 for k in keys] 
        input_features.append(feats)
    
    # Inference batch size = 1
    input_tensor = torch.tensor([input_features], dtype=torch.float).to(device)
    
    model = PointerNetwork(
        input_dim=config["input_dim"], 
        hidden_dim=config["hidden_dim"],
        d_model=config.get("d_model", 256)
    ).to(device)
    
    model_path = Path(config["model_folder"]) / "best_ptr.pt"
    if not model_path.exists():
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        _, pointers = model(input_tensor) 
        
    sort_indices = pointers.squeeze(0).cpu().tolist()
    
    with open(input_path, "r") as f:
        clean_json = json.load(f)
        
    sorted_data = [clean_json[i] for i in sort_indices]
    
    with open(config["output_json"], "w") as f:
        json.dump(sorted_data, f, indent=4)
        
    print(f"Saved to {config['output_json']}")

if __name__ == "__main__":
    run_pointer()
