# CODE 34.0: Severity Triage Setup with FINALIZED Conversational Aliases

import pandas as pd
import joblib

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

# --- 2. DEFINE FINAL ALIASES (CRITICAL EXPANSION) ---

MASTER_ALIAS_MAP = {
    # Key in CSV -> List of Aliases (Cleaned)
    'high_fever': ['fever', 'high_temp', 'feverish', 'hot'],
    'headache': ['head_pain', 'head_ache', 'head_soreness'],
    'dizziness': ['dizzy', 'faint'],
    'stomach_pain': ['stomach_ache', 'gut_pain', 'belly_ache'],
    'joint_pain': ['joints_hurt', 'joint_sore', 'knee_pain', 'hip_joint_pain', 'joint_ache'],
    
    # *** FIX FOR MUSCLE SORENESS ***
    'muscle_pain': ['muscle_soreness', 'muscle_ache', 'body_pain', 'body_ache', 'sore_muscles'], 
    
    # Breathing/Cold Fixes
    'breathlessness': ['breathing_issue', 'short_breath', 'breath_shortness', 'trouble_breathing'], 
    'congestion': ['stuffy_nose', 'blocked_nose', 'nose_block', 'nasal_block'], 
    'chills': ['cold', 'chilly', 'shivering'],
    'throat_irritation': ['throat_soreness', 'sore_throat', 'sore_throat_feeling'],
}

# --- 3. BUILD FINAL WEIGHT MAP ---
FINAL_WEIGHT_MAP = {}
# Add all official symptoms and weights first
for official_symptom, weight in symptom_weight_map.items():
    FINAL_WEIGHT_MAP[official_symptom] = weight

# Add all aliases, linking them back to the official symptom's weight
for official_symptom, aliases in MASTER_ALIAS_MAP.items():
    if official_symptom in FINAL_WEIGHT_MAP:
        weight = FINAL_WEIGHT_MAP[official_symptom]
        for alias in aliases:
            # We add the alias with the official symptom's weight
            FINAL_WEIGHT_MAP[alias] = weight 

# --- 4. ADJUSTED Triage Thresholds (Unchanged) ---
TRIAGE_THRESHOLDS = {
    'LOW_MAX': 5,      
    'MODERATE_MAX': 10 
}
joblib.dump(TRIAGE_THRESHOLDS, 'triage_thresholds.pkl')

# --- 5. Save the New Weight Map ---
joblib.dump(FINAL_WEIGHT_MAP, 'symptom_weight_map.pkl')

print(f"\nTotal weighted symptoms mapped: {len(FINAL_WEIGHT_MAP)}")
print("Severity Triage setup complete. Map updated with conversational aliases.")