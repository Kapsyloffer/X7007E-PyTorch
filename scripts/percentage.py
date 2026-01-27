import json
from scripts.overlaps import overlaps

def percentage():
    predicted = "jsons/predicted.json"
    shuffled = "jsons/shuffled.json"

    predicted_count = overlaps(predicted)
    shuffled_count = overlaps(shuffled)
    
    denominator = max(1, shuffled_count) 
    
    ratio = predicted_count / denominator

    ratio_true = ratio * 100
    ratio_percentage = (1-ratio) * 100

    print(f"Predicted: {predicted_count}")
    print(f"Shuffled:  {shuffled_count}")
    print(f"Ratio (pred/shuff): {ratio_true:.2f}%")
    print(f"Improvement: {ratio_percentage:.2f}%")
    return ratio_percentage

percentage()
