# CODE 35.0: Multilingual Translation Bridge Setup (Hindi/Tamil)

import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- 1. Define Language Codes and Model ---
# We will use the mBART model, highly effective for Indic languages.
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
LANG_HINDI_CODE = "hi_IN"
LANG_TAMIL_CODE = "ta_IN"
LANG_ENGLISH_CODE = "en_XX"

# --- 2. Load Tokenizer and Model ---
print(f"1. Loading Multilingual Model: {MODEL_NAME}...")
try:
    # Set device to GPU if available for faster operation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    
except Exception as e:
    print(f"\nFATAL ERROR: Failed to load Transformer model. Details: {e}")
    print("Ensure 'transformers' and 'torch' are installed and you have network access.")
    exit()

# --- 3. Translation Function ---
def translate_text(text, src_lang, tgt_lang):
    """Translates text between specified languages."""
    
    # Set source language for the tokenizer
    tokenizer.src_lang = src_lang
    
    # Encode and generate translation
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate translation (max_length prevents short, cut-off sentences)
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=150
    )
    
    # Decode and return
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


# --- 4. Save Translation Components ---
# We save the model object and tokenizer object
joblib.dump(tokenizer, 'multilingual_tokenizer.pkl')
joblib.dump(model, 'multilingual_translator.pkl')

print("\nMultilingual Bridge Setup Complete.")
print("Saved translator and tokenizer for Hindi/Tamil/English.")