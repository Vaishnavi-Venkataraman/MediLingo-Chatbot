
import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# --- 1. Data Loading and Preparation ---
print("1. Loading Training and Testing Data...")
try:
    df_train = pd.read_csv('Training.csv') 
    df_test = pd.read_csv('Testing.csv')   
except FileNotFoundError:
    print("\nERROR: Please ensure 'Training.csv' and 'Testing.csv' are in your folder.")
    exit()

if 'Unnamed: 133' in df_train.columns:
    df_train = df_train.drop('Unnamed: 133', axis=1)

# X: Features (Symptoms) and Y: Target (Disease)
X_train = df_train.drop('prognosis', axis=1)
X_test = df_test.drop('prognosis', axis=1)
y_train_text = df_train['prognosis']
y_test_text = df_test['prognosis']

# Feature Engineering: Target Label Encoding
le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_test = le.transform(y_test_text)

SYMPTOM_LIST = [col.strip().replace(' ', '_').lower() for col in X_train.columns]
X_train.columns = SYMPTOM_LIST 
X_test.columns = SYMPTOM_LIST


print(f"Training Data Shape (X, y): {X_train.shape}, {y_train.shape}")
print(f"Total Unique Diseases (Classes): {len(le.classes_)}")

print("\n2. Training Logistic Regression Model (Rectified OVR)...")
lr_model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=500, random_state=42))
lr_model.fit(X_train, y_train)

joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\n3. Model (lr_model.pkl) and Encoder (label_encoder.pkl) saved successfully.")

def preprocess_custom_symptoms(custom_symptoms: list, symptom_list: list) -> pd.DataFrame:
    """
    Creates the required binary feature vector (1x132 matrix) with correct feature names.
    """
    input_series = pd.Series(0, index=symptom_list)

    cleaned_symptoms = [s.strip().replace(' ', '_').lower() for s in custom_symptoms]

    recognized_symptoms = []
    for symptom in cleaned_symptoms:
        if symptom in symptom_list:
            input_series.loc[symptom] = 1
            recognized_symptoms.append(symptom)
        else:
            print(f"Warning: Symptom '{symptom}' not recognized by the model dictionary.")
    return input_series.to_frame().T 


def predict_disease(model, encoder, symptom_input_df, top_k=3):
    """Predicts the disease, confidence, and top k alternatives."""

    if symptom_input_df.iloc[0].sum() == 0:
        return "No Recognized Symptoms Provided", []

    prediction_index = model.predict(symptom_input_df)[0]
    predicted_disease = encoder.inverse_transform([prediction_index])[0]
    
    # 2. Probability Score 
    probabilities = model.predict_proba(symptom_input_df)[0]
    
    # Get the top K predictions
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    
    results = []
    for i in top_k_indices:
        disease = encoder.inverse_transform([i])[0]
        confidence = probabilities[i]
        results.append((disease, confidence))
        
    return predicted_disease, results

# --- 5. Custom Test Cases (Test 'shortness_of_breath' again) ---
# NOTE: The original dataset uses 'shortness_of_breath'. 
# Our cleaning step (replace(' ', '_')) should fix the user's previous input.
test_symptoms_1 = ['high_fever', 'headache', 'vomiting', 'diarrhoea', 'nausea'] 
# We fix the input format by using the cleaned, underscored name
test_symptoms_2 = ['chest_pain', 'sweating', 'vomiting', 'shortness_of_breath'] 
test_symptoms_3 = ['skin_rash', 'fatigue', 'patches_in_throat', 'high_fever'] 

# --- 6. Execute Test Cases ---
print("\n\n--- 4. Testing Custom Input 1 ---")
input_df_1 = preprocess_custom_symptoms(test_symptoms_1, SYMPTOM_LIST)
predicted_1, top_results_1 = predict_disease(lr_model, le, input_df_1)

print(f"Input Symptoms: {', '.join(test_symptoms_1)}")
print(f"PREDICTED DISEASE: **{predicted_1}**")
print("Top 3 Predictions:")
for disease, confidence in top_results_1:
    print(f"  - {disease}: {confidence*100:.2f}% Confidence")
    
print("\n--- Testing Custom Input 2 (Rectified) ---")
input_df_2 = preprocess_custom_symptoms(test_symptoms_2, SYMPTOM_LIST)
predicted_2, top_results_2 = predict_disease(lr_model, le, input_df_2)

print(f"Input Symptoms: {', '.join(test_symptoms_2)}")
print(f"PREDICTED DISEASE: **{predicted_2}**")
print("Top 3 Predictions:")
for disease, confidence in top_results_2:
    print(f"  - {disease}: {confidence*100:.2f}% Confidence")