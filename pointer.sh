rm -rf ML/weights_ptr
python scripts/new_json.py 
python ML/PointerNetwork/train.py
python ML/PointerNetwork/run.py
python -m scripts.percentage
#cp jsons/predicted.json jsons/predicted_old.json
#python scripts/refit.py jsons/predicted.json 
#python -m scripts.percentage
python scripts/verify.py
