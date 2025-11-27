import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.losses import MultipleNegativesRankingLoss
import torch
from tqdm.autonotebook import tqdm

try:
    X_train_text = pd.read_csv('X_train_text.csv')['symptoms_text']
    y_train_id = np.load('y_train_id.npy')
    le = joblib.load('le_semantic.pkl')
except FileNotFoundError:
    print("\nERROR: Data files not found.")
    exit()

NEW_BASE_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
try:
    sbert_model = SentenceTransformer(NEW_BASE_MODEL)
except Exception as e:
    print(f"FATAL ERROR: Failed to load SBERT model. Details: {e}")
    exit()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
sbert_model.to(device)

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

train_dataset = SentencesDataset(train_examples, model=sbert_model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16) 
train_loss = MultipleNegativesRankingLoss(sbert_model)

sbert_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='sbert_model_deep_finetuned',
    show_progress_bar=True,
    save_best_model=True
)

joblib.dump(sbert_model, 'sbert_model_deep_finetuned.pkl')