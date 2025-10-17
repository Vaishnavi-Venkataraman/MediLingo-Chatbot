# CODE 12.1: FINAL Chatbot Interface (Integrated Triage Fix + Top 3 + Description)

import pandas as pd
import numpy as np
import joblib
import re
from sentence_transformers import SentenceTransformer 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# --- 1. Load All Components ---
print("1. Loading FINAL Chatbot Components...")
try:
    # Prediction Components 
    svm_semantic_model = joblib.load('svm_semantic_model_final.pkl') 
    le = joblib.load('le_semantic.pkl')
    # Use the name saved during the final training step (Code 7.1)
    vectorizer_model = joblib.load('sbert_model_final.pkl') 
    
    # Triage Components (Requires Code 10.0)
    symptom_weight_map = joblib.load('symptom_weight_map.pkl')
    triage_thresholds = joblib.load('triage_thresholds.pkl')

    # NEW: Disease Description Map (Requires Code 12.0)
    disease_description_map = joblib.load('disease_description_map.pkl')

    # Text cleaning function (must be consistent with training)
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: One or more required files are missing. Details: {e}")
    print("Ensure all setup scripts ran successfully.")
    exit()

print("All components loaded. Starting Chatbot.")


# --- 2. Triage Scoring Function (Fixed Logic) ---
def calculate_triage_score(symptoms_text: str, weight_map: dict) -> tuple:
    """Calculates the total severity score and classifies risk based on thresholds."""
    total_score = 0
    recognized_symptoms = []
    # Prepare the search string by cleaning and using underscores to match map keys
    search_string = symptoms_text.lower().replace(' ', '_')
    
    for symptom_phrase, weight in weight_map.items():
        if symptom_phrase in search_string:
            total_score += weight
            recognized_symptoms.append(f"{symptom_phrase} (Weight: {weight})")
            
    # Determine Triage Level based on new thresholds
    if total_score > triage_thresholds['MODERATE_MAX']:
        triage_level = "🔴 HIGH RISK (Score > 10)"
        triage_advice = "Seek **IMMEDIATE** professional medical attention."
    elif total_score > triage_thresholds['LOW_MAX']:
        triage_level = "🟡 MODERATE RISK (Score 6-10)"
        triage_advice = "Consult a doctor within 24-48 hours."
    else:
        triage_level = "🟢 LOW RISK (Score 1-5)"
        triage_advice = "Monitor symptoms and consider self-care/FAQ advice."
        
    return total_score, triage_level, triage_advice, recognized_symptoms


# --- 3. Main Chatbot Loop ---
def run_chatbot(model, vectorizer, encoder, description_map):
    
    print("\n\n--- MEDILINGO AI CHATBOT (ENGLISH CORE) ---")
    print("**NO CONFIDENCE FILTER APPLIED.**")
    print("Type 'exit' to end.")

    while True:
        user_input = input("\nEnter your symptoms: ").strip()
        
        if user_input.lower() == 'exit':
            break
        if not user_input:
            continue
            
        # A. Semantic Prediction
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.encode([cleaned_input]) 
        probabilities = model.predict_proba(input_vector)[0]
        
        # --- Get Top 3 Predictions ---
        top_k_indices = np.argsort(probabilities)[::-1][:3]
        top_3_predictions = []
        for i in top_k_indices:
            disease = encoder.inverse_transform([i])[0]
            confidence = probabilities[i]
            top_3_predictions.append((disease, confidence))

        # B. Severity Triage
        triage_score, triage_level, triage_advice, recognized_symptoms = calculate_triage_score(user_input, symptom_weight_map)
        
        
        print("\n=============================================")
        print(f"         INTEGRATED CHATBOT RESPONSE (Total Score: {triage_score})")
        print("=============================================")
        
        # 1. Prediction Output
        print("--- 1. DISEASE PREDICTION (Top 3) ---")
        
        for i, (disease, confidence) in enumerate(top_3_predictions):
            priority = "Primary" if i == 0 else f"Alternative {i}"
            print(f"  {priority}: **{disease}** ({confidence*100:.2f}%)")
            
            # --- Display Description for the PRIMARY Prediction ---
            if i == 0:
                # Need to sanitize the disease name key for lookup
                key = disease.strip().lower().replace(' ', '_')
                description = description_map.get(key, "Description not found in map.")
                print(f"  Description: {description}")


        # 2. Triage Output
        print("\n--- 2. SEVERITY TRIAGE ---")
        print(f"Risk Level: {triage_level}")
        print(f"Recognized Severity Tokens: {', '.join(recognized_symptoms) if recognized_symptoms else 'None'}")
        print(f"**Recommended Action:** {triage_advice}")
        print("=============================================")
        
# --- 4. Start the Chatbot ---
if __name__ == '__main__':
    run_chatbot(svm_semantic_model, vectorizer_model, le, disease_description_map)