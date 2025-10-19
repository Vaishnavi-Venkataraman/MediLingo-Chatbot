import sys 
import pandas as pd
import numpy as np
import joblib
import re
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set the matplotlib backend for reliable pop-ups
plt.switch_backend('QtAgg') 

# --- GLOBAL MEMORY STORE & LANGUAGE CODES ---
LANG_HINDI_CODE = "hi_IN"
LANG_TAMIL_CODE = "ta_IN"
LANG_ENGLISH_CODE = "en_XX"
LANG_MAP = {'en': LANG_ENGLISH_CODE, 'hi': LANG_HINDI_CODE, 'ta': LANG_TAMIL_CODE, 
            LANG_ENGLISH_CODE: 'en', LANG_HINDI_CODE: 'hi', LANG_TAMIL_CODE: 'ta'}

CHAT_MEMORY = {'last_disease': None, 'last_lang': LANG_ENGLISH_CODE}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Load All Components ---
print("1. Loading FINAL Chatbot Components...")
try:
    # 1a. Prediction & Utility
    svm_semantic_model = joblib.load('svm_semantic_model_final.pkl') 
    le = joblib.load('le_semantic.pkl')
    vectorizer_model = joblib.load('sbert_model_final.pkl')
    
    # 1b. Triage & Description
    symptom_weight_map = joblib.load('symptom_weight_map.pkl')
    triage_thresholds = joblib.load('triage_thresholds.pkl')
    disease_description_map = joblib.load('disease_description_map.pkl')
    
    # 1c. FAQ & Multilingual Components (Removing Flan-T5 reliance)
    FAQ_EMBEDDINGS = np.load('faq_embeddings.npy')
    df_faq = pd.read_csv('faq_knowledge_base.csv')
    
    # NOTE: The translator model is sufficient for the final stable design.
    TRANSLATOR_MODEL = joblib.load('multilingual_translator.pkl').to(DEVICE)
    TRANSLATOR_TOKENIZER = joblib.load('multilingual_tokenizer.pkl')

    # Text cleaning function
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: One or more required files are missing. Details: {e}")
    exit()

print("All components loaded. Starting Chatbot.")

# ----------------------------------------------------------------------
# --- LANGUAGE & GENERATION FUNCTIONS ---
# ----------------------------------------------------------------------

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

# --- STATIC RAG Function (The only stable form) ---
def retrieve_faq_answer(user_input: str, model, embeddings, df_kb, context_disease=None, threshold=0.7):
    
    modified_query = user_input
    
    # 1. Contextual Substitution (its/this)
    if context_disease and ('its' in user_input.lower() or 'this' in user_input.lower()):
        modified_query = user_input.lower().replace('its', context_disease).replace('this', context_disease)

    # 2. CRITICAL FIX: Force the query to a common "What are the precautions for X?" structure.
    action_keywords = ['exercise', 'precautions', 'treatment', 'good for', 'avoid', 'what should i do']
    
    if context_disease and any(k in user_input.lower() for k in action_keywords):
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


# ----------------------------------------------------------------------
# --- CORE FEATURE FUNCTIONS (Abbreviated) ---
# ----------------------------------------------------------------------

def calculate_triage_score(symptoms_text: str, weight_map: dict) -> tuple:
    # Triage Logic (Final Fixed Version)
    recognized_symptoms_raw = []
    search_string = symptoms_text.lower().replace(' ', '_')
    recognized_keys = set()
    symptom_details = {} 
    for symptom_phrase, weight in weight_map.items():
        if symptom_phrase in search_string:
            if symptom_phrase not in recognized_keys:
                symptom_details[symptom_phrase] = weight
                recognized_keys.add(symptom_phrase)

    final_score = 0
    final_recognized_list = []
    official_weights_used = {} 

    for key, weight in symptom_details.items():
        if weight == 7 and 7 in official_weights_used: continue
        final_score += weight
        official_weights_used[weight] = key 
        final_recognized_list.append(f"{key} (Weight: {weight})")

    if final_score > triage_thresholds['MODERATE_MAX']:
        triage_level = "🔴 HIGH RISK (Score > 10)"
        triage_advice = "Seek **IMMEDIATE** professional medical attention."
    elif final_score > triage_thresholds['LOW_MAX']:
        triage_level = "🟡 MODERATE RISK (Score 6-10)"
        triage_advice = "Consult a doctor within 24-48 hours."
    else:
        triage_level = "🟢 LOW RISK (Score 1-5)"
        triage_advice = "Monitor symptoms and consider self-care/FAQ advice."
        
    return final_score, triage_level, triage_advice, final_recognized_list


def explain_prediction_scoreboard(top_3_predictions, recognized_symptoms_list):
    
    print("\n--- QUANTITATIVE EXPLANATION (SCOREBOARD) ---")
    
    symptom_data = {}
    for s_entry in recognized_symptoms_list:
        name = s_entry.split('(')[0].strip()
        weight_str = s_entry.split('Weight: ')[1].replace(')', '')
        symptom_data[name] = int(weight_str)

    if not symptom_data:
        print("No high-weight symptoms were recognized by Triage to form a clear explanation.")
        return
        
    total_recognized_weight = sum(symptom_data.values())
    
    print(f"Total Triage Weight driving prediction: {total_recognized_weight}")
    print("-------------------------------------------------")
    
    for disease_name, confidence in top_3_predictions:
        print(f"\n[Disease: {disease_name} ({confidence*100:.2f}%)]")
        
        for symptom, weight in symptom_data.items():
            weight_percent = (weight / total_recognized_weight) * 100 if total_recognized_weight > 0 else 0
            
            print(f"  - {symptom.replace('_', ' ').title(): <20}: Weight: {weight} ({weight_percent:.1f}% contribution to score)")


def visualize_multi_causal_links(top_3_predictions, recognized_symptoms_list):
    
    symptom_names = [s.split('(')[0].strip() for s in recognized_symptoms_list if s]

    if not symptom_names:
        return

    num_predictions = 3
    
    fig, axes = plt.subplots(1, num_predictions, figsize=(5 * num_predictions, 6))
    
    if num_predictions == 1: axes = [axes]

    for i, (disease, confidence) in enumerate(top_3_predictions):
        
        disease_node = disease.strip().lower().replace(' ', '_')
        subgraph = nx.DiGraph() 
        for symptom in symptom_names:
            subgraph.add_edge(symptom, disease_node)

        pos = nx.circular_layout(subgraph) 
        NODE_SIZE_NEW = 4000 
        FONT_SIZE_NEW = 7.5
        
        node_colors = ['#FFC300' if node == disease_node else '#7D3C98' for node in subgraph.nodes]
        node_labels = {node: node.replace('_', ' ').title() for node in subgraph.nodes}
        
        nx.draw(subgraph, pos, ax=axes[i], with_labels=True, labels=node_labels,
                node_size=NODE_SIZE_NEW, node_color=node_colors, font_size=FONT_SIZE_NEW, 
                font_color='white', arrowstyle='-|>', arrowsize=15)
        
        axes[i].set_title(f"{disease.title()}\nConf: {confidence*100:.1f}%", fontsize=10)

    plt.tight_layout()
    plt.show(block=False) 
    plt.pause(0.1) 
    print("\nMulti-Graph Window initiated. Check your taskbar/desktop.")

# ----------------------------------------------------------------------
# --- 7. MAIN CHATBOT LOOP (Multilingual & Generative RAG Disabled) ---
# ----------------------------------------------------------------------

def run_chatbot(model, vectorizer, encoder, description_map):
    
    print("\n\n--- MEDILINGO AI CHATBOT (MULTILINGUAL CORE) ---")
    print("Languages Supported: English, Hindi, Tamil (type 'exit' to end)")
    print("Mode: STABLE RETRIEVAL ONLY (Generative LLM Disabled)")

    while True:
        user_input = input("\nQuery/Symptoms (English, हिंदी, தமிழ்): ").strip()
        
        if user_input.lower() == 'exit': break
        if not user_input: continue
            
        # 1. LANGUAGE DETECTION
        src_lang_code = detect_language(user_input)
        src_lang_display = LANG_MAP.get(src_lang_code, 'English')
        CHAT_MEMORY['last_lang'] = src_lang_code
        print(f"| Language Detected: {src_lang_display} |")

        # 2. TRANSLATE INPUT TO ENGLISH (If needed)
        if src_lang_code != LANG_ENGLISH_CODE:
            user_input_en = translate_text(user_input, src_lang_code, LANG_ENGLISH_CODE)
        else:
            user_input_en = user_input
            
        # --- PHASE 1: ATTEMPT CONTEXTUAL FAQ RETRIEVAL (STATIC) ---
        context_disease_name = CHAT_MEMORY['last_disease'] if CHAT_MEMORY['last_disease'] else None

        match_q, match_a, score = retrieve_faq_answer(
            user_input_en, vectorizer_model, FAQ_EMBEDDINGS, df_faq, 
            context_disease=context_disease_name
        )
        
        if match_a:
            # SUCCESS: Static Retrieval Answer
            print("\n=============================================")
            print("         STATIC FAQ RETRIEVAL")
            print("=============================================")
            
            # Translate matched question and answer for display
            match_q_output = translate_text(match_q, LANG_ENGLISH_CODE, src_lang_code)
            match_a_output = translate_text(match_a, LANG_ENGLISH_CODE, src_lang_code)
            
            print(f"Query Matched (Score: {score:.3f}): {match_q_output}")
            print(f"**Answer ({src_lang_display}):**\n{match_a_output}")
            
            print("=============================================")
            continue 
        
        # --- PHASE 2: FALLBACK TO SYMPTOM CHECKER ---
        print("\n--- FALLBACK: SYMPTOM CHECKER ---")
        
        # A. Semantic Prediction
        cleaned_input = clean_text(user_input_en)
        input_vector = vectorizer_model.encode([cleaned_input]).astype(np.float64) 
        probabilities = model.predict_proba(input_vector)[0]
        
        top_k_indices = np.argsort(probabilities)[::-1][:3]
        top_3_predictions = []
        for i in top_k_indices:
            disease = encoder.inverse_transform([i])[0]
            confidence = probabilities[i]
            top_3_predictions.append((disease, confidence))
            
        primary_disease = top_3_predictions[0][0]
        CHAT_MEMORY['last_disease'] = primary_disease

        # B. Severity Triage
        triage_score, triage_level, triage_advice_en, recognized_symptoms = calculate_triage_score(user_input_en, symptom_weight_map)
        
        # 3. TRANSLATE OUTPUTS
        if src_lang_code != LANG_ENGLISH_CODE:
            primary_disease_output = translate_text(primary_disease, LANG_ENGLISH_CODE, src_lang_code)
            triage_advice_output = translate_text(triage_advice_en, LANG_ENGLISH_CODE, src_lang_code)
        else:
            primary_disease_output = primary_disease
            triage_advice_output = triage_advice_en

        print("\n=============================================")
        print(f"         INTEGRATED CHATBOT RESPONSE (Total Score: {triage_score})")
        print("=============================================")
        
        # 1. Prediction Output
        print("--- 1. DISEASE PREDICTION (Top 3) ---")
        
        for i, (disease, confidence) in enumerate(top_3_predictions):
            priority = "Primary" if i == 0 else f"Alternative {i}"
            
            display_disease = primary_disease_output if i == 0 else disease
            
            print(f"  {priority} ({src_lang_display}): **{display_disease}** ({confidence*100:.2f}%)")
            
            if i == 0:
                key = disease.strip().lower().replace(' ', '_')
                description = disease_description_map.get(key, "Description not found in map.")
                description_output = translate_text(description, LANG_ENGLISH_CODE, src_lang_code)
                print(f"  Description ({src_lang_display}): {description_output}")


        # 2. Triage Output
        print("\n--- 2. SEVERITY TRIAGE ---")
        print(f"Risk Level: {triage_level}")
        print(f"Recognized Severity Tokens: {', '.join(recognized_symptoms) if recognized_symptoms else 'None'}")
        print(f"**Recommended Action ({src_lang_display}):** {triage_advice_output}")
        
        # 3. Quantitative Scoreboard
        explain_prediction_scoreboard(top_3_predictions, recognized_symptoms)

        # 4. Multi-Graph Visualization (Call the function)
        visualize_multi_causal_links(top_3_predictions, recognized_symptoms)

        print("=============================================")

if __name__ == '__main__':
        run_chatbot(svm_semantic_model, vectorizer_model, le, disease_description_map)