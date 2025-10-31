import pandas as pd
import numpy as np
import re
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer 
import os 

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

# --- 2. Save Components (CRITICAL STEP) ---
print("\n2. Saving data splits and encoder...")

try:
    # Save Data Splits
    X_train_text.to_csv('X_train_text.csv', index=False, header=True)
    X_test_text.to_csv('X_test_text.csv', index=False, header=True)
    np.save('y_train_id.npy', y_train_id.values)
    np.save('y_test_id.npy', y_test_id.values)
    joblib.dump(le, 'le_semantic.pkl')
    
    # --- Check for successful file creation ---
    if os.path.exists('X_train_text.csv'):
        print("SUCCESS: Data files created. Proceeding to model building.")
    else:
        print("FATAL: File creation failed (Permission issue?). Stopping.")
        exit()

except Exception as e:
    print(f"FATAL ERROR during file saving: {e}")
    exit()

# --- 3. Semantic Vectorization (Code 4.2) ---
print("\n--- 3. Semantic Vectorization (Embedding Generation) ---")
MODEL_NAME = 'all-MiniLM-L6-v2'
sbert_model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings for Training data...")
X_train_vectors = sbert_model.encode(X_train_text.tolist(), show_progress_bar=True)
print("Generating embeddings for Testing data...")
X_test_vectors = sbert_model.encode(X_test_text.tolist(), show_progress_bar=True)

# --- Save Initial Vectors and SBERT Model ---
np.save('X_train_vectors.npy', X_train_vectors)
np.save('X_test_vectors.npy', X_test_vectors)
joblib.dump(sbert_model, 'sbert_model.pkl')

print("Semantic Vectorization complete. Initial model saved.")


# --- 4. Training Semantic SVM Classifier (Baseline) ---
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("\n--- 4. Training Semantic SVM Classifier ---")
svm_semantic_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_semantic_model.fit(X_train_vectors, y_train_id)

# Evaluation
y_pred_semantic = svm_semantic_model.predict(X_test_vectors)
accuracy_semantic = accuracy_score(y_test_id, y_pred_semantic)

print("\n--- Evaluation Results (Semantic SVM) ---")
print(f"Accuracy on Test Data: **{accuracy_semantic*100:.2f}%**")
joblib.dump(svm_semantic_model, 'svm_semantic_model_baseline.pkl') 

print("--------------------------------------------------")
print("Foundation script SUCCESS. Now run fine_tune_semantic_model.py")