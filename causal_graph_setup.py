import pandas as pd
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork

try:
    df_train = pd.read_csv('Training.csv')
except FileNotFoundError:
    print("\nERROR: Please ensure 'Training.csv' is in your folder.")
    exit()

if 'Unnamed: 133' in df_train.columns:
    df_train = df_train.drop('Unnamed: 133', axis=1)

df_train = df_train.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
df_train = df_train.fillna(0)

disease_node = 'prognosis'
symptom_nodes = [col for col in df_train.columns if col != disease_node]
edges = [(symptom, disease_node) for symptom in symptom_nodes]

causal_model_structure = DiscreteBayesianNetwork(edges)

joblib.dump(causal_model_structure, 'causal_model.pkl')
joblib.dump(disease_node, 'disease_node.pkl')
joblib.dump(symptom_nodes, 'symptom_nodes.pkl')