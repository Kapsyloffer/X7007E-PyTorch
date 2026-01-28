python scripts/new_json.py 
python -m ML.seq.train
python -m ML.seq.run
python -m scripts.percentage
#cp jsons/predicted.json jsons/predicted_old.json
#python scripts/refit.py jsons/predicted.json 
#python -m scripts.percentage
python scripts/verify.py
python -m scripts.check_constraints
