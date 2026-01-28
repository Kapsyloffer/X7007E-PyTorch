from pathlib import Path

def get_obj_config():
    objects = 100
    stations = 2
    takt = 700
    drift = 200
    gap = 5
    multiplier = 3
    min_size = 100
    max_size = takt + 2*drift
    return {
        "objects": objects,
        "stations": stations,
        "takt": takt,
        "drift": drift,
        "gap": gap,
        "training_multiplier": multiplier,
        "min_size": min_size,
        "max_size": max_size
# max_size = takt + 2 * drift_area

    }

