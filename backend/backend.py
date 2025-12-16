from flask import Flask, jsonify, request
from scripts.overlaps import *
import tempfile
import json
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"data": "hello world"})

@app.route("/check", methods=["POST"])
def disp():
    data = request.get_json()
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=True)

    json.dump(data, temp_file)
    temp_file.flush()
    temp_file.seek(0)
    path = temp_file.name
    
    result = overlaps(path)
    
    return jsonify({"overlap_count": result})

app.run(debug=True)
