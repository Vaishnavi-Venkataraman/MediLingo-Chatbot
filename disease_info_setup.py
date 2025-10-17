# CODE 12.0 (Corrected): Disease Description Setup

import pandas as pd
import joblib

print("1. Loading Disease Description Data...")
try:
    # NOTE: Using the provided file name
    df_desc = pd.read_csv('Symptom_Description.csv') 
except FileNotFoundError:
    print("\nERROR: Please ensure 'Symptom_Description.csv' is in your folder.")
    exit()

# --- 2. Robust Column Identification and Standardization ---
# Rename columns to lowercase for easy access
df_desc.columns = [col.strip().lower() for col in df_desc.columns]

# The correct column names based on your input:
DISEASE_NAME_COLUMN = 'disease' 
DESCRIPTION_COLUMN = 'description'

# --- 3. Create and Save the Description Map ---
print(f"Disease Name Column identified as: '{DISEASE_NAME_COLUMN}'")
print(f"Description Column identified as: '{DESCRIPTION_COLUMN}'")

try:
    # Create dictionary: Key = standardized disease name, Value = Description text
    # Standardize the keys (disease names) to match the LabelEncoder output (lowercase, underscore separation)
    
    # First, convert the dataframe structure to a dictionary
    raw_map = df_desc.set_index(DISEASE_NAME_COLUMN)[DESCRIPTION_COLUMN].to_dict()
    
    # Then, standardize the keys for matching (e.g., 'Drug Reaction' -> 'drug_reaction')
    standardized_map = {k.strip().lower().replace(' ', '_'): v for k, v in raw_map.items()}

except KeyError:
    print("\nFATAL ERROR: Column names not found after initial lowercasing. Check CSV headers.")
    exit()

joblib.dump(standardized_map, 'disease_description_map.pkl')

print(f"\nTotal disease descriptions mapped: {len(standardized_map)}")
print("Disease Description setup complete. Map saved as 'disease_description_map.pkl'.")