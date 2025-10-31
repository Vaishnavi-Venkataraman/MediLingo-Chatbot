# CODE 73.0: Severity Triage Setup with Comprehensive Conversational Aliases

import pandas as pd
import joblib
import re

# --- 1. Load Symptom Weights Dataset ---
print("1. Loading Symptom-Severity Mapping Data...")
try:
    df_weights = pd.read_csv('Symptom-severity.csv') 
except FileNotFoundError:
    print("\nERROR: Please ensure 'Symptom-severity.csv' is in your folder.")
    exit()

# Preprocessing and Mapping
df_weights.columns = [col.strip().replace(' ', '_').lower() for col in df_weights.columns]
symptom_weight_map = df_weights.set_index('symptom')['weight'].to_dict()

# --- 2. DEFINE FINAL ALIASES (CRITICAL, COMPREHENSIVE EXPANSION) ---
# NOTE: Aliases must be clean (lowercase, underscores instead of spaces/commas)

MASTER_ALIAS_MAP = {
    # Core Symptoms (High Frequency)
    'high_fever': ['fever', 'high_temp', 'feverish', 'hot', 'temperature'],
    'chills': ['cold', 'chilly', 'shivering'],
    'headache': ['head_pain', 'head_ache', 'head_soreness'],
    'vomiting': ['puking', 'throwing_up', 'emesis'],
    'diarrhoea': ['loose_motions', 'diarrhea', 'runs'],
    'cough': ['coughing'],
    'fatigue': ['tiredness', 'exhaustion', 'always_tired'],
    'joint_pain': ['joints_hurt', 'joint_sore', 'joint_ache', 'sore_joints'],
    'muscle_pain': ['muscle_soreness', 'muscle_ache', 'body_pain', 'sore_muscles'],
    'stomach_pain': ['stomach_ache', 'gut_pain', 'belly_ache', 'abdominal_cramps'],
    'chest_pain': ['heart_pain', 'chest_ache'],
    'skin_rash': ['rashes', 'skin_eruptions', 'skin_irritation'],
    
    # Urinary Symptoms
    'burning_micturition': ['burning_urine', 'painful_urination', 'burning_when_i_pee'],
    'spotting__urination': ['blood_in_urine', 'spotting_after_pee'],
    'bladder_discomfort': ['pee_discomfort', 'bladder_pain'],
    'continuous_feel_of_urine': ['constant_urge_to_pee'],
    'foul_smell_of_urine': ['bad_smelling_urine'],
    
    # Liver/Digestive
    'yellowish_skin': ['jaundice', 'skin_yellow'],
    'yellowing_of_eyes': ['yellow_eyes'],
    'dark_urine': ['pee_dark'],
    'indigestion': ['heartburn'],
    'stomach_bleeding': ['bleeding_stomach'],
    
    # Thyroid/Obesity
    'weight_gain': ['gaining_weight'],
    'weight_loss': ['losing_weight'],
    'puffy_face_and_eyes': ['swollen_face', 'puffy_eyes'],
    'enlarged_thyroid': ['swollen_neck'],
    'obesity': ['overweight'],
    
    # Neurological/Circulatory
    'dizziness': ['dizzy', 'faint', 'fainting'],
    'spinning_movements': ['vertigo', 'spinning'],
    'loss_of_balance': ['unstable', 'cannot_balance'],
    'weakness_of_one_body_side': ['paralysis_one_side'],
    'blurred_and_distorted_vision': ['blurry_vision'],
    'fast_heart_rate': ['palpitations', 'heart_pounding'],
    
    # Respiratory
    'breathlessness': ['breathing_issue', 'short_breath', 'trouble_breathing'],
    'throat_irritation': ['throat_soreness', 'sore_throat', 'throat_pain'],
    'congestion': ['stuffy_nose', 'blocked_nose', 'nose_block', 'nasal_block'],
    
    # Skin/Topical
    'nodal_skin_eruptions': ['skin_lumps', 'skin_bumps'],
    'red_sore_around_nose': ['sore_nose'],
    'yellow_crust_ooze': ['oozing_sore'],
}

# --- 3. BUILD FINAL WEIGHT MAP ---
FINAL_WEIGHT_MAP = {}
for official_symptom, weight in symptom_weight_map.items():
    # Add the official symptom itself (ensures the original keys are there)
    FINAL_WEIGHT_MAP[official_symptom] = weight
    
    # Add aliases, linking them back to the official symptom's weight
    aliases = MASTER_ALIAS_MAP.get(official_symptom, [])
    for alias in aliases:
        FINAL_WEIGHT_MAP[alias] = weight 

# --- 4. ADJUSTED Triage Thresholds (Unchanged) ---
TRIAGE_THRESHOLDS = {
    'LOW_MAX': 5,      
    'MODERATE_MAX': 10 
}
joblib.dump(TRIAGE_THRESHOLDS, 'triage_thresholds.pkl')

# --- 5. Save the New Weight Map ---
joblib.dump(FINAL_WEIGHT_MAP, 'symptom_weight_map.pkl')

print("Severity Triage setup complete. Map updated with all conversational aliases.")