import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
LANG_HINDI_CODE = "hi_IN"
LANG_TAMIL_CODE = "ta_IN"
LANG_ENGLISH_CODE = "en_XX"

print(f"1. Loading Multilingual Model: {MODEL_NAME}...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    
except Exception as e:
    print(f"\nFATAL ERROR: Failed to load Transformer model. Details: {e}")
    print("Ensure 'transformers' and 'torch' are installed and you have network access.")
    exit()

def translate_text(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=150
    )
    
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
joblib.dump(tokenizer, 'multilingual_tokenizer.pkl')
joblib.dump(model, 'multilingual_translator.pkl')