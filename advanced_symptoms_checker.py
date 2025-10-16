# CODE 4.1-4.3 (Unified): Semantic SVM Symptoms Checker (Robust English Model)

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# Ensure this dependency is installed: pip install sentence-transformers
from sentence_transformers import SentenceTransformer 

# --- 1. Data Preparation, Cleaning, and Setup (Code 4.1) ---
print("--- 1. Data Preparation and Cleaning ---")
try:
    df_text = pd.read_csv('Symptom2Disease.csv') 
except FileNotFoundError:
    print("\nERROR: Please ensure 'Symptom2Disease.csv' is in your folder.")
    exit()

df_text.columns = df_text.columns.str.lower()
df_text = df_text.rename(columns={'label': 'disease', 'text': 'symptoms_text'})
df_text = df_text.dropna(subset=['symptoms_text', 'disease'])

if 'index' in df_text.columns:
    df_text = df_text.drop('index', axis=1)

def clean_text(text):
    """Clean the symptom descriptions."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_text['symptoms_text'] = df_text['symptoms_text'].apply(clean_text)

le = LabelEncoder()
df_text['disease_id'] = le.fit_transform(df_text['disease'])

X_train_text, X_test_text, y_train_id, y_test_id = train_test_split(
    df_text['symptoms_text'], 
    df_text['disease_id'], 
    test_size=0.2, 
    random_state=42,
    stratify=df_text['disease_id']
)

joblib.dump(le, 'le_semantic.pkl')
print(f"Total Unique Diseases: {len(le.classes_)}")
print("Data preparation complete.")


# --- 2. Semantic Vectorization (Code 4.2) ---
print("\n--- 2. Semantic Vectorization (Embedding Generation) ---")

# Choose and Load Sentence Transformer Model
MODEL_NAME = 'all-MiniLM-L6-v2'
print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
try:
    sbert_model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"ERROR: Failed to load SentenceTransformer. Did you run 'pip install sentence-transformers'? Details: {e}")
    exit()

# Encode Data (Generate Embeddings)
print("Generating embeddings for Training data...")
X_train_vectors = sbert_model.encode(X_train_text.tolist(), show_progress_bar=True)
print("Generating embeddings for Testing data...")
X_test_vectors = sbert_model.encode(X_test_text.tolist(), show_progress_bar=True)

# Save the Sentence Transformer model separately for the interactive script
joblib.dump(sbert_model, 'sbert_model.pkl')

print("Semantic Vectorization complete.")


# --- 3. Semantic SVM Classification (Code 4.3) ---
print("\n--- 3. Training Semantic SVM Classifier ---")

# Set probability=True to enable predict_proba, needed for confidence scoring
svm_semantic_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_semantic_model.fit(X_train_vectors, y_train_id)

# Evaluation
y_pred_semantic = svm_semantic_model.predict(X_test_vectors)
accuracy_semantic = accuracy_score(y_test_id, y_pred_semantic)

print("\n--- Evaluation Results (Semantic SVM) ---")
print(f"Accuracy on Test Data: **{accuracy_semantic*100:.2f}%**")

# Save Semantic Model
joblib.dump(svm_semantic_model, 'svm_semantic_model.pkl')

print("\nSemantic SVM Model saved as 'svm_semantic_model.pkl'.")
print("--------------------------------------------------")
print("Ready for Interactive Testing (Code 4.4).")