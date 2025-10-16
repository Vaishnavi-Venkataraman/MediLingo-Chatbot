# CODE 10.1: Final Chatbot Interface (Integrated Triage, Fixed Error, New Thresholds)

import pandas as pd
import numpy as np
import joblib
import re
from sentence_transformers import SentenceTransformer 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# --- 1. Load All Components (Prediction and Triage) ---
print("1. Loading FINAL Chatbot Components...")
try:
    # 1a. Prediction Components 
    svm_semantic_model = joblib.load('svm_semantic_model_final.pkl') 
    le = joblib.load('le_semantic.pkl')
    vectorizer_model = joblib.load('sbert_model_final.pkl')
    
    # 1b. Triage Components
    symptom_weight_map = joblib.load('symptom_weight_map.pkl')
    triage_thresholds = joblib.load('triage_thresholds.pkl')

    # Text cleaning function (must be consistent with training)
    def clean_text(text):
        """Cleans text for Semantic (SBERT) vectorization."""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: One or more required files are missing. Details: {e}")
    print("Please ensure you ran all setup scripts: Code 7.0/7.1 (Training) AND Code 10.0 (Triage Setup).")
    exit()

print("All components loaded. Starting Chatbot.")


# --- 2. Triage Scoring Function (Using New Thresholds) ---
def calculate_triage_score(symptoms_text: str, weight_map: dict) -> tuple:
    """
    Calculates the total severity score and classifies risk based on new thresholds.
    """
    total_score = 0
    recognized_symptoms = []
    
    # Prepare the search string by cleaning and using underscores to match map keys (Fuzzy Logic)
    search_string = symptoms_text.lower().replace(' ', '_')
    
    for symptom_phrase, weight in weight_map.items():
        if symptom_phrase in search_string:
            total_score += weight
            recognized_symptoms.append(f"{symptom_phrase} (Weight: {weight})")
            
    # Determine Triage Level based on new thresholds (1-5 Low, 6-10 Moderate, >10 High)
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
def run_chatbot(model, vectorizer, encoder):
    """Handles the user interaction, prediction, and triage output without a confidence threshold."""
    
    print("\n\n--- MEDILINGO AI CHATBOT (ENGLISH CORE) ---")
    print("**NO CONFIDENCE FILTER APPLIED.** Top prediction shown regardless of score.")
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
        prediction_index = model.predict(input_vector)[0]
        predicted_disease = encoder.inverse_transform([prediction_index])[0]
        top_confidence = probabilities[prediction_index]
        
        # B. Severity Triage
        # Pass the raw user input to the triage function for specialized keyword searching
        triage_score, triage_level, triage_advice, recognized_symptoms = calculate_triage_score(user_input, symptom_weight_map)
        
        print("\n=============================================")
        print(f"         INTEGRATED CHATBOT RESPONSE (Total Score: {triage_score})")
        print("=============================================")
        
        # 1. Prediction Output
        print("--- 1. DISEASE PREDICTION ---")
        print(f"**Potential Condition:** {predicted_disease}")
        print(f"Confidence: {top_confidence*100:.2f}%")
        
        # 2. Triage Output
        print("\n--- 2. SEVERITY TRIAGE ---")
        print(f"Risk Level: {triage_level}")
        print(f"Recognized Severity Tokens: {', '.join(recognized_symptoms) if recognized_symptoms else 'None'}")
        print(f"**Recommended Action:** {triage_advice}")
        print("=============================================")
        
# --- 4. Start the Chatbot ---
if __name__ == '__main__':
    run_chatbot(svm_semantic_model, vectorizer_model, le)