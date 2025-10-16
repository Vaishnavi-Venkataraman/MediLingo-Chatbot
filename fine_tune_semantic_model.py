# CODE 7.0: Deep Fine-Tuning using a Multilingual Base Model (3 Epochs)

import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.losses import MultipleNegativesRankingLoss
import torch
from tqdm.autonotebook import tqdm

# --- 1. Load Data and Components ---
print("1. Loading Data and Components for Deep Fine-Tuning...")
try:
    X_train_text = pd.read_csv('X_train_text.csv')['symptoms_text']
    y_train_id = np.load('y_train_id.npy')
    le = joblib.load('le_semantic.pkl')
except FileNotFoundError:
    print("\nERROR: Data files not found. Please run Code 4.1 first.")
    exit()

# --- 2. Initialize NEW Base Model (Multilingual & Robust) ---
NEW_BASE_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
print(f"2. Initializing NEW multilingual base model: {NEW_BASE_MODEL}...")
try:
    sbert_model = SentenceTransformer(NEW_BASE_MODEL)
except Exception as e:
    print(f"FATAL ERROR: Failed to load SBERT model. Details: {e}")
    exit()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
sbert_model.to(device)


# --- 3. Create Training Examples ---
print("3. Preparing Contrastive Training Data...")
train_examples = []
text_list = X_train_text.tolist()
label_list = y_train_id.tolist()

text_by_label = {}
for text, label in zip(text_list, label_list):
    if label not in text_by_label:
        text_by_label[label] = []
    text_by_label[label].append(text)

for label, texts in text_by_label.items():
    if len(texts) >= 2:
        for i in range(len(texts) - 1):
             train_examples.append(InputExample(texts=[texts[i], texts[i+1]]))
        train_examples.append(InputExample(texts=[texts[-1], texts[0]]))


# --- 4. Setup DataLoader and Loss Function ---
train_dataset = SentencesDataset(train_examples, model=sbert_model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16) 
train_loss = MultipleNegativesRankingLoss(sbert_model)


# --- 5. DEEP FINE-TUNING (Increased Epochs) ---
print(f"\n4. Starting DEEP Fine-Tuning ({NEW_BASE_MODEL}, 3 Epochs)...")
sbert_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3, # Increased epochs for better specialization
    warmup_steps=100,
    output_path='sbert_model_deep_finetuned',
    show_progress_bar=True,
    save_best_model=True
)

# --- 6. Save the Fine-Tuned Model ---
joblib.dump(sbert_model, 'sbert_model_deep_finetuned.pkl')
print("\n5. Deep Fine-Tuned SBERT Model saved as 'sbert_model_deep_finetuned.pkl'.")
print("Ready to retrain the SVM.")