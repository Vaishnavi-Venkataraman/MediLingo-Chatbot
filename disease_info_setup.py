import pandas as pd
import joblib

try:
    df_desc = pd.read_csv('Symptom_Description.csv') 
except FileNotFoundError:
    print("\nERROR: Please ensure 'Symptom_Description.csv' is in your folder.")
    exit()
df_desc.columns = [col.strip().lower() for col in df_desc.columns]

DISEASE_NAME_COLUMN = 'disease' 
DESCRIPTION_COLUMN = 'description'

try:
    raw_map = df_desc.set_index(DISEASE_NAME_COLUMN)[DESCRIPTION_COLUMN].to_dict()
    standardized_map = {k.strip().lower().replace(' ', '_'): v for k, v in raw_map.items()}

except KeyError:
    print("\nFATAL ERROR: Column names not found after initial lowercasing. Check CSV headers.")
    exit()

joblib.dump(standardized_map, 'disease_description_map.pkl')

print(f"\nTotal disease descriptions mapped: {len(standardized_map)}")