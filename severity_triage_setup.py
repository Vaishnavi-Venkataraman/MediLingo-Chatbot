import pandas as pd
import joblib

print("1. Loading Symptom-Severity Mapping Data...")
try:
    df_weights = pd.read_csv('Symptom-severity.csv') 
except FileNotFoundError:
    print("\nERROR: Please ensure 'Symptom-severity.csv' is in your folder.")
    exit()

df_weights.columns = [col.strip().replace(' ', '_').lower() for col in df_weights.columns]
symptom_weight_map = df_weights.set_index('symptom')['weight'].to_dict()

MASTER_ALIAS_MAP = {
    'high_fever': ['fever', 'high_temp', 'feverish', 'hot'],
    'headache': ['head_pain', 'head_ache', 'head_soreness'],
    'dizziness': ['dizzy', 'faint'],
    'stomach_pain': ['stomach_ache', 'gut_pain', 'belly_ache'],
    'joint_pain': ['joints_hurt', 'joint_sore', 'knee_pain', 'hip_joint_pain', 'joint_ache'],
    'muscle_pain': ['muscle_soreness', 'muscle_ache', 'body_pain', 'body_ache', 'sore_muscles'], 
    'breathlessness': ['breathing_issue', 'short_breath', 'breath_shortness', 'trouble_breathing'], 
    'congestion': ['stuffy_nose', 'blocked_nose', 'nose_block', 'nasal_block'], 
    'chills': ['cold', 'chilly', 'shivering'],
    'throat_irritation': ['throat_soreness', 'sore_throat', 'sore_throat_feeling'],
}

FINAL_WEIGHT_MAP = {}
for official_symptom, weight in symptom_weight_map.items():
    FINAL_WEIGHT_MAP[official_symptom] = weight

for official_symptom, aliases in MASTER_ALIAS_MAP.items():
    if official_symptom in FINAL_WEIGHT_MAP:
        weight = FINAL_WEIGHT_MAP[official_symptom]
        for alias in aliases:
            FINAL_WEIGHT_MAP[alias] = weight 

TRIAGE_THRESHOLDS = {
    'LOW_MAX': 5,      
    'MODERATE_MAX': 10 
}
joblib.dump(TRIAGE_THRESHOLDS, 'triage_thresholds.pkl')

joblib.dump(FINAL_WEIGHT_MAP, 'symptom_weight_map.pkl')

print(f"\nTotal weighted symptoms mapped: {len(FINAL_WEIGHT_MAP)}")
print("Severity Triage setup complete. Map updated with conversational aliases.")