import streamlit as st
import requests
import pandas as pd

# Flask server URL
FLASK_API_URL = "http://127.0.0.1:5000/predict"

# Load symptoms from CSV
file_path = "./SYMTOMS.xlsx"
df = pd.read_excel(file_path)
symptom_columns = df.drop(columns=["prognosis"]).columns.tolist()

# Streamlit UI
st.title("Disease Prediction System ")
st.write("Select 'Yes' for symptoms you have and get a predicted disease.")

# Display symptoms in 4 columns
columns = st.columns(2)
selected_symptoms = []

for index, symptom in enumerate(symptom_columns):
    col = columns[index % 2]  # Distribute symptoms across 4 columns
    if col.checkbox(symptom, False):
        selected_symptoms.append(symptom)

if st.button("Predict Disease"):
    response = requests.post(FLASK_API_URL, json={"symptoms": selected_symptoms})
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Disease: **{result['predicted_disease']}**")
    else:
        st.error("Error in prediction. Please check server logs.")
