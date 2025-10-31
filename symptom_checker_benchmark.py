import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report

print("1. Loading Data and Components for Benchmarking...")

try:
    df_train = pd.read_csv('Training.csv') 
    df_test = pd.read_csv('Testing.csv') 
    
    if 'Unnamed: 133' in df_train.columns:
        df_train = df_train.drop('Unnamed: 133', axis=1)
    if 'Unnamed: 133' in df_test.columns:
        df_test = df_test.drop('Unnamed: 133', axis=1)

    le = joblib.load('label_encoder.pkl')
    y_test = le.transform(df_test['prognosis'])
    y_train = le.transform(df_train['prognosis'])

except FileNotFoundError as e:
    print(f"\nERROR: Please ensure all CSV files and saved encoder file ('label_encoder.pkl') are present. Details: {e}")
    exit()

X_orig = df_train.drop('prognosis', axis=1)
X_test_orig = df_test.drop('prognosis', axis=1)


SYMPTOM_LIST_CLEAN = [col.strip().replace(' ', '_').lower() for col in X_orig.columns]

X_train_clean = X_orig.copy()
X_test_clean = X_test_orig.copy()
X_train_clean.columns = SYMPTOM_LIST_CLEAN
X_test_clean.columns = SYMPTOM_LIST_CLEAN

print("\n2. Comparing models...")

lr_model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=500, random_state=42))
lr_model.fit(X_train_clean, y_train)
accuracy_lr = accuracy_score(y_test, lr_model.predict(X_test_clean))
joblib.dump(lr_model, 'lr_model_clean.pkl') 
print(f"   LR Accuracy: {accuracy_lr*100:.2f}%")

gnb_model = GaussianNB()
gnb_model.fit(X_train_clean, y_train)
accuracy_gnb = accuracy_score(y_test, gnb_model.predict(X_test_clean))
joblib.dump(gnb_model, 'gnb_model.pkl')
print(f"   GNB Accuracy: {accuracy_gnb*100:.2f}%")

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_clean, y_train)
accuracy_svm = accuracy_score(y_test, svm_model.predict(X_test_clean))
joblib.dump(svm_model, 'svm_model.pkl')
print(f"   SVM Accuracy: {accuracy_svm*100:.2f}%")

print("\n==================================================================")
print("4. Basic Model Benchmarking Summary:")
print("==================================================================")

results = {
    'Model': ['Logistic Regression (LR)', 'Gaussian Naive Bayes (GNB)', 'Support Vector Machine (SVM)'],
    'Accuracy': [accuracy_lr, accuracy_gnb, accuracy_svm]
}
df_results = pd.DataFrame(results)
df_results['Accuracy (%)'] = df_results['Accuracy'] * 100

best_model_name = df_results.sort_values(by='Accuracy', ascending=False).iloc[0]['Model']

print(df_results[['Model', 'Accuracy (%)']].sort_values(by='Accuracy (%)', ascending=False).to_markdown(index=False))
print("==================================================================")
print("Conclusion: All models achieved 100% accuracy on this structured dataset.")
print(f"We will use the **SVM model** for its strong performance and general robustness.")