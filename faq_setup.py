# CODE 27.0: Medical FAQ Setup and Semantic Embedding (Finalized)

import pandas as pd
import numpy as np
import joblib
import re
from sentence_transformers import SentenceTransformer 

# --- 1. Load Data and Components ---
print("1. Loading required data and Sentence-Transformer model...")
try:
    # Load the Precaution dataset
    df_raw = pd.read_csv('symptom_precaution.csv') # Adjust name to your actual file name
    
    # Load the FINAL Sentence Transformer model for consistent vector creation
    sbert_model = joblib.load('sbert_model_final.pkl')
    
    # Text cleaning function from the main script for consistency
    def clean_text(text):
        """Cleans text for Semantic (SBERT) vectorization."""
        text = text.lower()
        # Remove special characters, preserving text content for semantic analysis
        text = re.sub(r'[^a-z\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Required file missing. Check your CSV file name and 'sbert_model_final.pkl'. Details: {e}")
    exit()

# --- 2. Prepare FAQ Knowledge Base ---
print("2. Structuring Q&A pairs...")

# Standardize columns to lowercase
df_raw.columns = [col.strip().lower().replace(' ', '_') for col in df_raw.columns]

# The main disease column to use as the source of the question
DISEASE_COL = 'disease' 

# The precaution columns to concatenate for the answer
PRECAUTION_COLS = [f'precaution_{i}' for i in range(1, 5)]

FAQ_DATA = []

for index, row in df_raw.iterrows():
    disease_name = row.get(DISEASE_COL, '').strip().title() # Title case for display in question
    
    # 2a. Generate the Question
    question = f"What are the precautions for {disease_name}?"
    cleaned_question = clean_text(question)
    
    # 2b. Concatenate the Answer (Precaution List)
    answers = []
    for col in PRECAUTION_COLS:
        answer = row.get(col, '')
        if pd.notna(answer) and answer:
            answers.append(str(answer).strip())

    if answers:
        # Join precautions into a numbered list string for the final answer
        final_answer = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
        
        FAQ_DATA.append({
            'question': cleaned_question,
            'answer': final_answer,
            'source_disease': disease_name
        })

df_faq = pd.DataFrame(FAQ_DATA).drop_duplicates(subset=['question'])
print(f"Total unique Q&A pairs created: {len(df_faq)}")


# --- 3. Generate Semantic Embeddings for all Questions ---
print("3. Generating embeddings for FAQ questions...")

question_list = df_faq['question'].tolist()

# Use the loaded SBERT model to create vectors (768 dimensions)
faq_embeddings = sbert_model.encode(question_list, show_progress_bar=True)

np.save('faq_embeddings.npy', faq_embeddings)
df_faq.to_csv('faq_knowledge_base.csv', index=False)

print("\nFAQ setup complete.")
print("Saved 'faq_embeddings.npy' and 'faq_knowledge_base.csv'.")