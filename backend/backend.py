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

# /run/transformer/<int> -> [förslag] (len = <int>)
@app.route("/run/transformer", methods=["POST"])
def api_run_transformer():
    data = request.get_json()
    path = temp_path(data)
    return run_transformer(path)

@app.route("/run/transformer/<int:nm>", methods=["POST"])
def api_run_transformers(nm):
    data = request.get_json()
    path = temp_path(data)
    förslag = []
    for x in range(0, nm):
        förslag.append(run_transformer(path))
    return förslag

@app.route("/jumpin/<str:obj>", methods=["POST"])
def api_jumpin(json, idx, obj): # JSON, target index, object
    data = request.get_json()
    path = temp_path(data)
    return "TODO"

@app.route("/jumpout/<str:obj>", methods=["POST"])
def api_jumpout(json, idx, obj): # JSON, target index, object
    data = request.get_json()
    path = temp_path(data)
    return "TODO"

# (KLIENT/LOCALSTORAGE) 1. Endpoint för ändringar i sekvensen. (Splitta Update till Uppdatera operatörer på en station, eller lägg till runners, eller flytta ordningen på produkter i sekvensen)
#
# 2. Endpoint för borttagning i sekvensen.
# 3. Endpoint för tilläggning av ordrar i sekvensen.
#
# (OPTIONAL) 4. Lagra sekvenser. (Endpoint för save, update, delete sequence)
#
# 5. Endpoint för att föreslå bättre* sekvenser baserat på data.
# * Definieras med färre violations.



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
