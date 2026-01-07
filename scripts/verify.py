import json
import sys
import math
from pathlib import Path

def smart_compare_dicts(d1, d2):
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    if keys1 != keys2:
        return False, f"Keys differ: {keys1} vs {keys2}"

    for k in keys1:
        v1 = d1[k]
        v2 = d2[k]

        # Check if values are effectively the same number
        try:
            val1 = float(v1)
            val2 = float(v2)
            if not math.isclose(val1, val2, rel_tol=1e-9, abs_tol=1e-9):
                return False, f"Value mismatch at {k}: {v1} vs {v2}"
        except (ValueError, TypeError):
            # If not numbers (unlikely for sizes), fall back to strict equality
            if v1 != v2:
                return False, f"Non-numeric mismatch at {k}: {v1} vs {v2}"
    
    return True, ""

def load_object_map(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        try:
            content = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {path}: {e}")
            sys.exit(1)

    if not isinstance(content, list):
        print(f"❌ Root element in {path} must be a list.")
        sys.exit(1)

    obj_map = {}
    for idx, item in enumerate(content):
        obj_id = item.get("object")
        # Store strictly 'data', default to empty dict
        obj_map[obj_id] = item.get("data", {})

    # print(f"✅ Loaded {len(obj_map)} objects from {path.name}")
    return obj_map

def verify_integrity():
    files = [
        #"jsons/allocations.json",
        "jsons/shuffled.json",
        "jsons/predicted.json"
    ]

    maps = {f: load_object_map(f) for f in files}
    
    base_file = files[0]
    base_map = maps[base_file]
    base_ids = set(base_map.keys())

    all_good = True

    # Compare others against the base
    for compare_file in files[1:]:
        compare_map = maps[compare_file]
        compare_ids = set(compare_map.keys())
        
        # 1. Check ID sets
        missing = base_ids - compare_ids
        extra = compare_ids - base_ids

        if missing:
            # print(f"   ❌ Missing Object IDs in {compare_file}: {missing}")
            all_good = False
        if extra:
            # print(f"   ❌ Extra Object IDs in {compare_file}: {extra}")
            all_good = False

        if missing or extra:
            continue 

        # 2. Smart Check Data Content
        data_mismatches = 0
        for obj_id in base_ids:
            base_data = base_map[obj_id]
            comp_data = compare_map[obj_id]

            # Use smart comparison instead of '=='
            is_same, reason = smart_compare_dicts(base_data, comp_data)
            
            if not is_same:
                data_mismatches += 1
                if data_mismatches <= 5: 
                    continue
                    # print(f"   ❌ Data mismatch for Object '{obj_id}':")
                    # print(f"      Reason: {reason}")
        
        if data_mismatches != 0:
            all_good = False

    print("\n" + "="*40)
    if all_good:
        print("🎉 INTEGRITY CHECK PASSED: All files are consistent.")
    else:
        print("mw INTEGRITY CHECK FAILED: Data corruption detected.")
    print("="*40)

if __name__ == "__main__":
    verify_integrity()
