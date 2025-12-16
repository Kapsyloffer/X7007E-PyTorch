from flask import Flask, jsonify, request
from scripts.overlaps import *
from ML.Transformer.run import *
import tempfile
import json
import os

app = Flask(__name__)

def temp_path(data):
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)

    json.dump(data, temp_file)
    temp_file.flush()
    temp_file.seek(0)
    return temp_file.name


@app.route("/", methods=["GET"])
def home():
    return jsonify({"data": "hello world"})

@app.route("/check", methods=["POST"])
def check():
    data = request.get_json()
    path = temp_path(data)
    result = overlaps(path)
    
    return jsonify({"overlap_count": result})

@app.route("/run/transformer", methods=["POST"])
def api_run_transformer():
    data = request.get_json()
    path = temp_path(data)
    return run_transformer(path)

@app.route("/run/pointernetwork", methods=["POST"])
def api_run_pointernetwork():
    data = request.get_json()
    path = temp_path(data)
    return "TODO"

@app.route("/run/seq2seq", methods=["POST"])
def api_run_seq2seq():
    data = request.get_json()
    path = temp_path(data)
    return "TODO"

app.run(debug=True)
