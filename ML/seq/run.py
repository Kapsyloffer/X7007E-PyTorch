import json
import torch
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ML.seq.model import build_seq2seq
from ML.seq.config import get_config

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_mmal_offsets(rearranged_objects):
    TAKT = 700 #FIXME
    DRIFT = 200
    station_keys = [f"s{i}" for i in range(1, 36)] #FIXME
    
    station_ready_time = {k: 0 for k in station_keys}
    
    for obj in rearranged_objects:
        obj["offsets"] = {}
        for i, s_key in enumerate(station_keys):
            size = obj["data"][s_key]
            
            congestion_bound = station_ready_time[s_key] - TAKT
            
            if i > 0:
                prev_s = station_keys[i-1]
                flow_bound = obj["offsets"][prev_s] + obj["data"][prev_s] - TAKT
            else:
                flow_bound = -DRIFT
            
            earliest_start = max(congestion_bound, flow_bound)
            
            min_offset = -DRIFT
            max_offset = TAKT - size + DRIFT
            
            actual_offset = max(min_offset, min(max_offset, earliest_start))
            
            obj["offsets"][s_key] = int(actual_offset)
            station_ready_time[s_key] = actual_offset + size
            
    return rearranged_objects

def run_seq():
    # Instantiating model using factory function
    # Note: output_dim set to 100 to match dataset max_seq_len/sorting indices
    model = build_seq2seq(
        input_dim=config["input_dim"], 
        hidden_dim=config["hidden_dim"], 
        output_dim=100, 
        dropout=config["dropout"],
        device=device
    )
    
    model_path = Path(config["model_folder"]) / "best_model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: {model_path} not found. Running with initialized weights.")
        
    model.eval()
    
    with open(config["run_path"], "r") as f:
        data = json.load(f)

    objects_to_sort = data[0] if isinstance(data[0], list) else data
    keys = [f"s{i}" for i in range(1, 36)] #FIXME
    features = [[obj["data"][k] / 1000.0 for k in keys] for obj in objects_to_sort]
    
    input_tensor = torch.tensor([features], dtype=torch.float).to(device)

    with torch.no_grad():
        # Pass tgt=None for inference
        _, predictions = model(input_tensor, tgt=None)
        predicted_indices = predictions.squeeze(0).cpu().numpy()

    # Create rearranged list based on predictions
    # Deduplicate indices to ensure valid permutation
    seen = set()
    unique_indices = []
    for idx in predicted_indices:
        if idx < len(objects_to_sort) and idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)
            
    # Append missing indices if any
    for i in range(len(objects_to_sort)):
        if i not in seen:
            unique_indices.append(i)
    
    rearranged = [objects_to_sort[idx] for idx in unique_indices]
    optimized_rearranged = calculate_mmal_offsets(rearranged)

    output_path = config.get("output_json", "predicted_ordered_seq.json")
    with open(output_path, "w") as f:
        json.dump(optimized_rearranged, f, indent=4)

    print(f"Reordered and offset-optimized sequence saved to {output_path}")

if __name__ == "__main__":
    run_seq()
