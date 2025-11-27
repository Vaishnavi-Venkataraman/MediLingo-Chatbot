import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer 
from sklearn.preprocessing import LabelEncoder

try:
    sbert_model_finetuned = joblib.load('sbert_model_deep_finetuned.pkl')

    X_train_text = pd.read_csv('X_train_text.csv')['symptoms_text']
    X_test_text = pd.read_csv('X_test_text.csv')['symptoms_text']
    y_train_id = np.load('y_train_id.npy')
    y_test_id = np.load('y_test_id.npy')
    
except FileNotFoundError:
    print("\nERROR: Ensure 'sbert_model_deep_finetuned.pkl is available")
    exit()

X_train_vectors_new = sbert_model_finetuned.encode(X_train_text.tolist(), show_progress_bar=True)
X_test_vectors_new = sbert_model_finetuned.encode(X_test_text.tolist(), show_progress_bar=True)

print("\n4. Training new SVC on specialized vectors...")
svm_semantic_model_new = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_semantic_model_new.fit(X_train_vectors_new, y_train_id)

y_pred_semantic_new = svm_semantic_model_new.predict(X_test_vectors_new)
accuracy_semantic_new = accuracy_score(y_test_id, y_pred_semantic_new)

print("\n--- Evaluation Results (Deep Fine-Tuned Semantic SVM) ---")
print(f"New Accuracy on Test Data: **{accuracy_semantic_new*100:.2f}%**")

joblib.dump(svm_semantic_model_new, 'svm_semantic_model_final.pkl')
joblib.dump(sbert_model_finetuned, 'sbert_model_final.pkl')
print("\nFinal Semantic SVM Model saved as 'svm_semantic_model_final.pkl'.")
print("Ready for final interactive testing and Triage setup.")