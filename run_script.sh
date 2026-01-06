source .env/bin/activate
python scripts/new_json.py
cp jsons/shuffled.json jsons/predicted.json
python -m scripts.percentage
python scripts/refit.py jsons/shuffled.json
python -m scripts.percentage
python scripts/visualize.py jsons/shuffled.json
