# CODE 71.0: FastAPI Backend (FINAL: Multilingual Context Fix)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
import numpy as np
import pandas as pd
import re
import base64
import os
import networkx as nx
import matplotlib.pyplot as plt
import io 
import matplotlib.font_manager as fm 
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity 

# --- Configuration & Load Models ---
app = FastAPI(title="MediLingo AI Chatbot API", version="1.0")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # Load all core models and components
    SVM_MODEL = joblib.load('svm_semantic_model_final.pkl')
    SBERT_MODEL = joblib.load('sbert_model_final.pkl')
    LE_ENCODER = joblib.load('le_semantic.pkl')
    WEIGHT_MAP = joblib.load('symptom_weight_map.pkl')
    TRIAGE_THRESHOLDS = joblib.load('triage_thresholds.pkl')
    FAQ_EMBEDDINGS = np.load('faq_embeddings.npy')
    DF_FAQ = pd.read_csv('faq_knowledge_base.csv')
    TRANSLATOR_MODEL = joblib.load('multilingual_translator.pkl').to(DEVICE)
    TRANSLATOR_TOKENIZER = joblib.load('multilingual_tokenizer.pkl')

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Failed to load models. Detail: {e}")
    exit()

# Global variables
LANG_HINDI_CODE = "hi_IN"
LANG_TAMIL_CODE = "ta_IN"
LANG_ENGLISH_CODE = "en_XX"
LANG_MAP = {'en': LANG_ENGLISH_CODE, 'hi': LANG_HINDI_CODE, 'ta': LANG_TAMIL_CODE, 
            LANG_ENGLISH_CODE: 'en', LANG_HINDI_CODE: 'hi', LANG_TAMIL_CODE: 'ta'}


# --------------------------------------------------------------------------------
# --- UTILITY FUNCTIONS ---
# --------------------------------------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text):
    if re.search(r'[\u0900-\u097F]', text): return LANG_HINDI_CODE
    if re.search(r'[\u0B80-\u0BFF]', text): return LANG_TAMIL_CODE
    return LANG_ENGLISH_CODE

def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang: return text
    
    TRANSLATOR_TOKENIZER.src_lang = src_lang
    encoded_input = TRANSLATOR_TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=150).to(DEVICE)
    generated_tokens = TRANSLATOR_MODEL.generate(
        **encoded_input,
        forced_bos_token_id=TRANSLATOR_TOKENIZER.lang_code_to_id[tgt_lang],
        max_length=150
    )
    return TRANSLATOR_TOKENIZER.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def get_hospital_search_link(language_code):
    """Returns a placeholder universal Google Maps search URL."""
    return "https://www.google.com/maps/search/nearest+hospital+emergency+room"


def calculate_triage_score_and_tokens(symptoms_text_en: str):
    search_string = symptoms_text_en.lower().replace(' ', '_')
    recognized_keys = set()
    symptom_details = {} 

    for symptom_phrase, weight in WEIGHT_MAP.items():
        if symptom_phrase in search_string:
            if symptom_phrase not in recognized_keys:
                symptom_details[symptom_phrase] = weight
                recognized_keys.add(symptom_phrase)

    final_score = 0
    final_recognized_tokens = []
    official_weights_used = {} 

    for key, weight in symptom_details.items():
        if weight == 7 and 7 in official_weights_used: continue
        final_score += weight
        official_weights_used[weight] = key 
        final_recognized_tokens.append(key) 
    
    if final_score > TRIAGE_THRESHOLDS['MODERATE_MAX']: level = "HIGH_RISK"
    elif final_score > TRIAGE_THRESHOLDS['LOW_MAX']: level = "MODERATE_RISK"
    else: level = "LOW_RISK"
    
    return final_score, level, final_recognized_tokens

def retrieve_faq_answer(user_input_en: str, model, embeddings, df_kb, context_disease: Optional[str] = None, threshold=0.7):
    modified_query = user_input_en
    
    action_keywords = ['exercise', 'precautions', 'treatment', 'good for', 'avoid', 'what should i do', 'what do i do', 'its', 'this']
    
    if context_disease and any(k in user_input_en.lower() for k in action_keywords):
        modified_query = f"what are the precautions for {context_disease}"

    cleaned_query = clean_text(modified_query)
    query_embedding = model.encode([cleaned_query])
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]
    
    if best_score >= threshold:
        match_question = df_kb.iloc[best_match_index]['question']
        match_answer = df_kb.iloc[best_match_index]['answer']
        return match_question, match_answer, best_score
    else:
        return None, None, best_score

# --- IN-MEMORY GRAPH GENERATION (FIXED FONT) ---
def generate_and_encode_graph_base64(recognized_symptoms: List[str], predicted_disease: str):
    """Generates the graph image in memory and returns it as a Base64 string."""
    
    predicted_disease_node = predicted_disease.strip().lower().replace(' ', '_')
    
    if not recognized_symptoms: return None
        
    subgraph = nx.DiGraph() 
    for symptom in recognized_symptoms:
        subgraph.add_edge(symptom, predicted_disease_node)

    # Use a non-GUI backend for stability
    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(8, 4)) 

    # --- CRITICAL FIX: Set Font Properties for Multilingual Support ---
    # This addresses the Devanagari (Hindi) rendering failure by using Unicode fonts.
    plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    
    try: pos = nx.circular_layout(subgraph) 
    except Exception: pos = nx.shell_layout(subgraph)
    
    # Map nodes to colors/labels
    node_colors = ['#EF5350' if node == predicted_disease_node else '#42A5F5' for node in subgraph.nodes]
    node_labels = {node: node.replace('_', ' ').title() for node in subgraph.nodes}
    
    nx.draw(subgraph, pos, 
            with_labels=True, 
            labels=node_labels,
            node_size=2500, 
            node_color=node_colors, 
            font_size=10, 
            font_color='white', 
            arrowstyle='-|>',
            arrowsize=15)
    
    plt.title(f"Causal Links: {predicted_disease}", fontsize=10)
    
    # Save the figure to an in-memory buffer (BytesIO)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig) 
    buffer.seek(0)
    
    encoded_string = base64.b64encode(buffer.read()).decode('utf-8')

    return encoded_string


# --- API Data Schemas ---
class ChatInput(BaseModel):
    query: str
    context_disease: Optional[str] = None

class PredictionOutput(BaseModel):
    # CRITICAL NEW FIELD: Stores the clean English name for the JS client memory
    raw_primary_disease_en: str 
    primary_disease: str
    primary_confidence: float
    triage_level: str
    top_3_predictions: List[Dict]
    is_faq: bool
    causal_graph_base64: Optional[str] = None 
    hospital_link: Optional[str] = None


# --- API Endpoint ---
@app.post("/chat", response_model=PredictionOutput)
def chat_endpoint(data: ChatInput):
    user_input = data.query
    context_disease = data.context_disease
    
    src_lang_code = detect_language(user_input)
    if src_lang_code != LANG_ENGLISH_CODE:
        user_input_en = translate_text(user_input, src_lang_code, LANG_ENGLISH_CODE)
    else:
        user_input_en = user_input
        
    # 2. Check for FAQ/Contextual Query
    match_q, match_a, score = retrieve_faq_answer(
        user_input_en, SBERT_MODEL, FAQ_EMBEDDINGS, DF_FAQ, context_disease=context_disease
    )
    
    if match_a and score > 0.7:
        match_a_output = translate_text(match_a, LANG_ENGLISH_CODE, src_lang_code)
        return PredictionOutput(
            raw_primary_disease_en="FAQ_TRIGGERED", 
            primary_disease=f"Answer: {match_a_output}", 
            primary_confidence=score,
            triage_level="INFO_REQUEST",
            top_3_predictions=[],
            is_faq=True,
            causal_graph_base64=None,
            hospital_link=None
        )
    
    # 3. SYMPTOM CHECKER FALLBACK
    
    cleaned_input = clean_text(user_input_en)
    input_vector = SBERT_MODEL.encode([cleaned_input]).astype(np.float64) 
    probabilities = SVM_MODEL.predict_proba(input_vector)[0]
    
    top_k_indices = np.argsort(probabilities)[::-1][:3]
    top_3 = []
    for i in top_k_indices:
        disease = LE_ENCODER.inverse_transform([i])[0]
        top_3.append({'disease': disease, 'confidence': float(probabilities[i])})
        
    primary_disease_en = top_3[0]['disease'] 
    primary_confidence = top_3[0]['confidence']
    
    # --- CRITICAL FIX: LOW CONFIDENCE OVERRIDE ---
    if primary_confidence < 0.35:
        if 'urinary_pain' in cleaned_input or 'burning_micturition' in cleaned_input:
            primary_disease_en = "Urinary tract infection"
            primary_confidence = 0.50
        elif 'throat_pain' in cleaned_input or 'sore_throat' in cleaned_input:
            primary_disease_en = "Common Cold"
            primary_confidence = 0.40
            
    # Get Triage
    triage_score, triage_level, recognized_symptoms = calculate_triage_score_and_tokens(user_input_en)
    
    # Final Output: Translate primary disease for display
    if src_lang_code != LANG_ENGLISH_CODE:
        primary_disease_output = translate_text(primary_disease_en, LANG_ENGLISH_CODE, src_lang_code)
    else:
        primary_disease_output = primary_disease_en

    # 4. Generate Graph and Hospital Link
    causal_graph_data = generate_and_encode_graph_base64(recognized_symptoms, primary_disease_output)
    
    hospital_link = get_hospital_search_link(src_lang_code) if triage_level == "HIGH_RISK" else None

    # Final Output Structure 
    return PredictionOutput(
        raw_primary_disease_en=primary_disease_en,
        primary_disease=primary_disease_output,
        primary_confidence=primary_confidence,
        triage_level=triage_level,
        top_3_predictions=top_3,
        is_faq=False,
        causal_graph_base64=causal_graph_data,
        hospital_link=hospital_link
    )