# CODE 10.0: Severity Triage Setup with Expanded Symptoms and NEW Thresholds

import pandas as pd
import joblib

# --- 1. Load Symptom Weights Dataset (Needed for Symptom-Weight Mapping) ---
try:
    df_weights = pd.read_csv('Symptom-severity.csv') 
except FileNotFoundError:
    print("\nERROR: Please ensure 'Symptom-severity.csv' is in your folder.")
    exit()

# Preprocessing and Mapping
df_weights.columns = [col.strip().replace(' ', '_').lower() for col in df_weights.columns]
symptom_weight_map = df_weights.set_index('symptom')['weight'].to_dict()

# --- 2. Expand Triage Map (Fuzzy Mapping for Safety) ---
# Adds common synonyms/variants from your experience:
symptom_weight_map['dizzy'] = symptom_weight_map.get('dizziness', 4) 
symptom_weight_map['head_pain'] = symptom_weight_map.get('headache', 3)
symptom_weight_map['stomach_ache'] = symptom_weight_map.get('stomach_pain', 5) 
symptom_weight_map['knee_pain'] = symptom_weight_map.get('joint_pain', 3) # Added knee pain

# --- 3. ADJUSTED Triage Thresholds (CRITICAL SAFETY FIX) ---
# New Requirements: 1-5 Low, 6-10 Moderate, >10 High
TRIAGE_THRESHOLDS = {
    'LOW_MAX': 5,      # Max score for Low Risk
    'MODERATE_MAX': 10 # Max score for Moderate Risk (Scores > 10 are High Risk)
}
joblib.dump(TRIAGE_THRESHOLDS, 'triage_thresholds.pkl')
print("Triage thresholds adjusted for safety and new scale.")

# --- 4. Save the New Weight Map ---
joblib.dump(symptom_weight_map, 'symptom_weight_map.pkl')

print(f"\nTotal weighted symptoms mapped: {len(symptom_weight_map)}")
print("Severity Triage setup complete. Maps saved.")