from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  

model = joblib.load("stress_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [
        data["Study_Hours_Per_Day"],
        data["Extracurricular_Hours_Per_Day"],
        data["Sleep_Hours_Per_Day"],
        data["Social_Hours_Per_Day"],
        data["Physical_Activity_Hours_Per_Day"],
        data["GPA"]
    ]

    X = np.array([features])
    pred = model.predict(X)[0]
    stress_label = label_encoder.inverse_transform([pred])[0]

    return jsonify({"stress_level": stress_label})

app.run(port=5000)
