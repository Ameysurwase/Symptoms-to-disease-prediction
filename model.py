import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = "C:\\model\\SYMTOMS.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Separate features and target variable
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "disease_prediction_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Predict on test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
def evaluate_model():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy * 100, precision * 100, recall * 100, f1 * 100

model_accuracy, model_precision, model_recall, model_f1 = evaluate_model()
print(f"Model Accuracy: {model_accuracy:.2f}%")
print(f"Model Precision: {model_precision:.2f}%")
print(f"Model Recall: {model_recall:.2f}%")
print(f"Model F1 Score: {model_f1:.2f}%")

# Function to predict disease
def predict_disease(user_symptoms):
    """
    Predicts the disease based on user input symptoms.
    
    Parameters:
        user_symptoms (list): List of symptoms (as column names) present in the user.
    
    Returns:
        str: Predicted disease.
    """
    # Load the trained model
    model = joblib.load("disease_prediction_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    
    # Create a feature vector initialized with zeros
    input_features = [0] * len(X.columns)
    
    # Set values to 1 for symptoms present in user input
    for symptom in user_symptoms:
        if symptom in X.columns:
            index = X.columns.get_loc(symptom)
            input_features[index] = 1
    
    # Predict using the trained model
    prediction = model.predict([input_features])
    
    # Decode the predicted label to disease name
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    
    return predicted_disease

# Take user input for symptoms
print("Answer 'yes' or 'no' for the following symptoms:")
user_symptoms = []
for symptom in X.columns:
    response = input(f"Do you have {symptom}? (yes/no): ").strip().lower()
    if response == "yes":
        user_symptoms.append(symptom)

# Predict and display the disease
predicted_disease = predict_disease(user_symptoms)
print(f"Predicted Disease: {predicted_disease}")
