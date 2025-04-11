from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load model only (no scaler)
with open("heart_model.pkl", "rb") as f:
    heart_model = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the Heart Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        features = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        if not all(feature in input_data for feature in features):
            return jsonify({"error": "Missing one or more required input features."}), 400

        input_values = [float(input_data[feature]) for feature in features]
        input_array = np.array([input_values])

        # No scaler used here
        prediction = heart_model.predict(input_array)

        result = "The person has Heart Disease" if prediction[0] == 1 else "The person does not have Heart Disease"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
