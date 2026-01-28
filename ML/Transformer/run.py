import json
import torch
import sys
import time
from pathlib import Path
from tqdm import tqdm  

from ML.Transformer.model import build_transformer 
from ML.Transformer.config import get_config
from scripts.config import get_obj_config

config = get_config()
obj_config = get_obj_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STN = obj_config["stations"] + 1
def calculate_mmal_offsets(rearranged_objects):
    TAKT = obj_config["takt"]
    DRIFT = obj_config["drift"] 
    station_keys = [f"s{i}" for i in range(1, STN)] 
    
    # Tidpunkt då stationen blir ledig för nästa objekt
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
            
            # -drift <= offset <= takt - size + drift
            min_offset = -DRIFT
            max_offset = TAKT - size + DRIFT
            
            actual_offset = max(min_offset, min(max_offset, earliest_start))
            
            obj["offsets"][s_key] = int(actual_offset)
            station_ready_time[s_key] = actual_offset + size
            
    return rearranged_objects

def run_transformer():
    model = build_transformer(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        N=config["n_layers"],
        h=config["n_heads"],
        dropout=0.0,
        d_ff=config["d_ff"]
    ).to(device)
    
    state_dict = torch.load(Path(config["model_folder"]) / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    with open(config["run_path"], "r") as f:
        data = json.load(f)

    objects_to_sort = data[0] if isinstance(data[0], list) else data
    keys = [f"s{i}" for i in range(1, STN)] 
    features = [[obj["data"][k] / 1000.0 for k in keys] for obj in objects_to_sort]
    
    input_tensor = torch.tensor([features], dtype=torch.float).to(device)
    B, S, F = input_tensor.shape
    
    print(f"Starting inference on {S} items using {device}...")
    
    torch.cuda.empty_cache()

    with torch.no_grad():
        src_mask = (input_tensor.abs().sum(dim=-1) > 0).unsqueeze(1).unsqueeze(2)
        encoder_output = model.encode(input_tensor, src_mask)
        
        decoder_input = model.sos_token.expand(B, 1, -1)
        predicted_indices = []
        
        start_t = time.time()
        
        iterator = tqdm(range(S), desc="Decoding", unit="item")
        
        for i in iterator:
            if i % 1000 == 0: 
                torch.cuda.empty_cache()

            dec_len = decoder_input.size(1)
            tgt_mask = torch.triu(torch.ones(1, dec_len, dec_len), diagonal=1).type(torch.bool).to(device) == 0
            
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            logits = model.project(decoder_output, encoder_output)
            
            # Greedy decode logic
            last_step_logits = logits[:, -1, :]
            if len(predicted_indices) > 0:
                last_step_logits[:, predicted_indices] = -float('inf')
            
            _, next_idx = torch.max(last_step_logits, dim=-1)
            next_idx_val = next_idx.item()
            predicted_indices.append(next_idx_val)
            
            next_feature = input_tensor[:, next_idx_val, :].unsqueeze(1)
            decoder_input = torch.cat([decoder_input, next_feature], dim=1)

    print(f"Done in {time.time() - start_t:.2f}s")
    
    rearranged = [objects_to_sort[idx] for idx in predicted_indices]
    optimized_rearranged = calculate_mmal_offsets(rearranged)

    output_path = config.get("output_json", "predicted_ordered.json")
    with open(output_path, "w") as f:
        json.dump(optimized_rearranged, f, indent=4)

    print(f"Reordered and offset-optimized sequence saved to {output_path}")

if __name__ == "__main__":
    run_transformer()
