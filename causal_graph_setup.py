# CODE 13.0 (Final Structural Fix): Causal Graph Setup (Visualization Only)

import pandas as pd
import joblib
import networkx as nx
import matplotlib.pyplot as plt
# Use the structural model class
from pgmpy.models import DiscreteBayesianNetwork 

# --- 1. Load Structured Training Data ---
print("1. Loading structured data for Causal Modeling...")
try:
    df_train = pd.read_csv('Training.csv') # Use the structured data
except FileNotFoundError:
    print("\nERROR: Please ensure 'Training.csv' is in your folder.")
    exit()

# Clean and prepare the data (ensure column names are simple keys)
if 'Unnamed: 133' in df_train.columns:
    df_train = df_train.drop('Unnamed: 133', axis=1)

df_train = df_train.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
df_train = df_train.fillna(0) # Fill NaN with 0

# --- 2. Define the Network Structure (Graph Components) ---
disease_node = 'prognosis'
symptom_nodes = [col for col in df_train.columns if col != disease_node]
edges = [(symptom, disease_node) for symptom in symptom_nodes]

print(f"Defining {len(symptom_nodes)} causal edges pointing to '{disease_node}'.")

# --- 3. Initialize and Save the Graph STRUCTURE ONLY ---
print("2. Initializing Causal Graph Structure (Skipping Training to prevent memory crash)...")

# Initialize the model structure (we save this empty structure for node/edge reference)
causal_model_structure = DiscreteBayesianNetwork(edges)

# Save the structural components
joblib.dump(causal_model_structure, 'causal_model.pkl')
joblib.dump(disease_node, 'disease_node.pkl')
joblib.dump(symptom_nodes, 'symptom_nodes.pkl')

print("Causal Graph STRUCTURE saved successfully.")
print("Auxiliary Causal Model components saved.")