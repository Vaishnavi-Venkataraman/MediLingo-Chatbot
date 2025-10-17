# CODE 20.0: Severity Triage Setup with Aggressive Conversational Mapping

import pandas as pd
import joblib

# --- 1. Define Triage Thresholds (Unchanged) ---
TRIAGE_THRESHOLDS = {
    'LOW_MAX': 5,      # Max score for Low Risk
    'MODERATE_MAX': 10 # Max score for Moderate Risk
}
joblib.dump(TRIAGE_THRESHOLDS, 'triage_thresholds.pkl')

# --- 2. Load Symptom Weights Dataset and Map ---
try:
    df_weights = pd.read_csv('Symptom-severity.csv') 
except FileNotFoundError:
    print("\nERROR: Please ensure 'Symptom-severity.csv' is in your folder.")
    exit()

df_weights.columns = [col.strip().replace(' ', '_').lower() for col in df_weights.columns]
symptom_weight_map = df_weights.set_index('symptom')['weight'].to_dict()

# --- 3. EXPANDED Triage Map (FINAL AGGRESSIVE ALIASES) ---
print("2. Expanding Triage Map with FINAL conversational aliases...")

MASTER_ALIAS_MAP = {
    # Key in CSV -> List of Aliases (Cleaned)
    'high_fever': ['fever', 'high_temp', 'feverish', 'hot'],
    'dizziness': ['dizzy', 'faint'],
    'headache': ['head_pain', 'head_ache'],
    'stomach_pain': ['stomach_ache', 'gut_pain'],
    'joint_pain': ['joints_hurt', 'joint_sore', 'knee_pain', 'hip_joint_pain'],
    
    # FIX: Breathing issues
    'breathlessness': ['breathing_issue', 'short_breath', 'breath_shortness', 'trouble_breathing', 'breathing_trouble'], 
    
    # FIX: Cold and congestion
    'congestion': ['stuffy_nose', 'blocked_nose', 'nose_block', 'nasal_block'], 
    'chills': ['cold', 'chilly', 'shivering'],
    'cough': ['coughing'],
    'throat_irritation': ['throat_soreness', 'sore_throat'],
}

FINAL_WEIGHT_MAP = {}
for official_symptom, weight in symptom_weight_map.items():
    FINAL_WEIGHT_MAP[official_symptom] = weight
    
    aliases = MASTER_ALIAS_MAP.get(official_symptom, [])
    for alias in aliases:
        FINAL_WEIGHT_MAP[alias] = weight 

# --- 4. Save the New Weight Map ---
joblib.dump(FINAL_WEIGHT_MAP, 'symptom_weight_map.pkl')

print(f"\nTotal weighted symptoms mapped: {len(FINAL_WEIGHT_MAP)}")
print("Severity Triage setup complete. Map updated with conversational aliases.")