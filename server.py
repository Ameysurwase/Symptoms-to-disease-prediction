from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model and label encoder
model = joblib.load("disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load symptom names from training data
file_path = r".\SYMTOMS.xlsx"  # Use raw string to avoid invalid escape sequence
df = pd.read_excel(file_path)  # Ensure openpyxl is installed
symptom_columns = df.drop(columns=["prognosis"]).columns.tolist()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get symptom data from request
        user_symptoms = data.get("symptoms", [])  # List of "yes" symptoms

        # Create a feature vector initialized with zeros
        input_features = [0] * len(symptom_columns)

        # Set values to 1 for symptoms present in user input
        for symptom in user_symptoms:
            if symptom in symptom_columns:
                index = symptom_columns.index(symptom)
                input_features[index] = 1

        # Predict using the trained model
        prediction = model.predict([input_features])
        predicted_disease = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"predicted_disease": predicted_disease})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
