import json
import torch
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ML.PointerNetwork.model import PointerNetwork
from ML.PointerNetwork.config import get_config

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_mmal_offsets(rearranged_objects):
    TAKT = 700 #FIXME
    DRIFT = 200
    station_keys = [f"s{i}" for i in range(1, 36)] #FIXME
    
    # Tidpunkt då stationen blir ledig för nästa objekt
    station_ready_time = {k: 0 for k in station_keys}
    
    for obj in rearranged_objects:
        obj["offsets"] = {}
        for i, s_key in enumerate(station_keys):
            size = obj["data"][s_key]
            
            # Gräns 1: När stationen blev ledig från föregående objekt
            congestion_bound = station_ready_time[s_key] - TAKT
            
            # Gräns 2: När föregående station på samma objekt blev klar (Ditt krav!)
            if i > 0:
                prev_s = station_keys[i-1]
                flow_bound = obj["offsets"][prev_s] + obj["data"][prev_s] - TAKT
            else:
                flow_bound = -DRIFT
            
            # Vi måste respektera den mest begränsande faktorn
            earliest_start = max(congestion_bound, flow_bound)
            
            # Hårda drift-begränsningar: -drift <= offset <= takt - size + drift
            min_offset = -DRIFT
            max_offset = TAKT - size + DRIFT
            
            actual_offset = max(min_offset, min(max_offset, earliest_start))
            
            obj["offsets"][s_key] = int(actual_offset)
            # Uppdatera när stationen är ledig för nästa objekt i kön
            station_ready_time[s_key] = actual_offset + size
            
    return rearranged_objects

def run_pointer():
    model = PointerNetwork(config["input_dim"], config["hidden_dim"], config["d_model"]).to(device)
    model.load_state_dict(torch.load(Path(config["model_folder"]) / "best_model.pt", map_location=device))
    model.eval()
    
    with open(config["run_path"], "r") as f:
        data = json.load(f)

    objects_to_sort = data[0] if isinstance(data[0], list) else data
    keys = [f"s{i}" for i in range(1, 36)] #FIXME
    features = [[obj["data"][k] / 1000.0 for k in keys] for obj in objects_to_sort]
    
    input_tensor = torch.tensor([features], dtype=torch.float).to(device)

    with torch.no_grad():
        _, pointers = model(input_tensor)
        predicted_indices = pointers.squeeze(0).cpu().numpy()

    rearranged = [objects_to_sort[idx] for idx in predicted_indices]
    optimized_rearranged = calculate_mmal_offsets(rearranged)

    output_path = config.get("output_json", "predicted_ordered.json")
    with open(output_path, "w") as f:
        json.dump(optimized_rearranged, f, indent=4)

    print(f"Reordered and offset-optimized sequence saved to {output_path}")

if __name__ == "__main__":
    run_pointer()
