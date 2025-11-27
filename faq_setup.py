import pandas as pd
import numpy as np
import joblib
import re
from sentence_transformers import SentenceTransformer 

try:
    df_raw = pd.read_csv('symptom_precaution.csv')
    sbert_model = joblib.load('sbert_model_final.pkl')

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Required file missing. Check your CSV file name and 'sbert_model_final.pkl'. Details: {e}")
    exit()

df_raw.columns = [col.strip().lower().replace(' ', '_') for col in df_raw.columns]

DISEASE_COL = 'disease' 
PRECAUTION_COLS = [f'precaution_{i}' for i in range(1, 5)]
FAQ_DATA = []

for index, row in df_raw.iterrows():
    disease_name = row.get(DISEASE_COL, '').strip().title()
    
    question = f"What are the precautions for {disease_name}?"
    cleaned_question = clean_text(question)

    answers = []
    for col in PRECAUTION_COLS:
        answer = row.get(col, '')
        if pd.notna(answer) and answer:
            answers.append(str(answer).strip())

    if answers:
        final_answer = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
        
        FAQ_DATA.append({
            'question': cleaned_question,
            'answer': final_answer,
            'source_disease': disease_name
        })

df_faq = pd.DataFrame(FAQ_DATA).drop_duplicates(subset=['question'])
print(f"Total unique Q&A pairs created: {len(df_faq)}")

question_list = df_faq['question'].tolist()
faq_embeddings = sbert_model.encode(question_list, show_progress_bar=True)

np.save('faq_embeddings.npy', faq_embeddings)
df_faq.to_csv('faq_knowledge_base.csv', index=False)