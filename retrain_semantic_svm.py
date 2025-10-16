# CODE 7.1: Retrain Semantic SVM on Deep Fine-Tuned Embeddings

import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer 
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Components and New Model ---
print("\n--- 1. Retraining SVM on Deep Fine-Tuned Embeddings ---")

try:
    # Load the DEEP fine-tuned SBERT model
    sbert_model_finetuned = joblib.load('sbert_model_deep_finetuned.pkl')
    # Load data for re-encoding
    X_train_text = pd.read_csv('X_train_text.csv')['symptoms_text']
    X_test_text = pd.read_csv('X_test_text.csv')['symptoms_text']
    y_train_id = np.load('y_train_id.npy')
    y_test_id = np.load('y_test_id.npy')
    
except FileNotFoundError:
    print("\nERROR: Ensure Code 7.0 ran successfully to generate 'sbert_model_deep_finetuned.pkl'.")
    exit()

# --- 2. Generate NEW Embeddings ---
print("2. Re-generating NEW embeddings for Training data...")
X_train_vectors_new = sbert_model_finetuned.encode(X_train_text.tolist(), show_progress_bar=True)
print("3. Re-generating NEW embeddings for Testing data...")
X_test_vectors_new = sbert_model_finetuned.encode(X_test_text.tolist(), show_progress_bar=True)


# --- 4. Model Training: Support Vector Machine (SVC) ---
print("\n4. Training new SVC on specialized vectors...")
svm_semantic_model_new = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_semantic_model_new.fit(X_train_vectors_new, y_train_id)

# --- 5. Evaluation ---
y_pred_semantic_new = svm_semantic_model_new.predict(X_test_vectors_new)
accuracy_semantic_new = accuracy_score(y_test_id, y_pred_semantic_new)

print("\n--- Evaluation Results (Deep Fine-Tuned Semantic SVM) ---")
print(f"New Accuracy on Test Data: **{accuracy_semantic_new*100:.2f}%**")

# --- 6. Save Final Model ---
joblib.dump(svm_semantic_model_new, 'svm_semantic_model_final.pkl')
joblib.dump(sbert_model_finetuned, 'sbert_model_final.pkl') # Save SBERT under a final name as well
print("\nFinal Semantic SVM Model saved as 'svm_semantic_model_final.pkl'.")
print("Ready for final interactive testing and Triage setup.")