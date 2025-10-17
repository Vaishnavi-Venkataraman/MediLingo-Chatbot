# CODE 27.0: FINAL Integrated Chatbot Interface (Node Sizing Fix for Visualization)

import pandas as pd
import numpy as np
import joblib
import re
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Set the matplotlib backend to a non-interactive one for reliable pop-ups
plt.switch_backend('QtAgg') 

# --- 1. Load All Components ---
print("1. Loading FINAL Chatbot Components...")
try:
    # 1a. Prediction Components 
    svm_semantic_model = joblib.load('svm_semantic_model_final.pkl') 
    le = joblib.load('le_semantic.pkl')
    vectorizer_model = joblib.load('sbert_model_final.pkl')
    
    # 1b. Triage Components
    symptom_weight_map = joblib.load('symptom_weight_map.pkl')
    triage_thresholds = joblib.load('triage_thresholds.pkl')

    # 1c. Disease Description Map
    disease_description_map = joblib.load('disease_description_map.pkl')
    
    # Text cleaning function
    def clean_text(text):
        """Cleans text for Semantic (SBERT) vectorization."""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: One or more required files are missing. Details: {e}")
    print("Please ensure all setup scripts ran successfully.")
    exit()

print("All components loaded. Starting Chatbot.")


# --- 2. Triage Scoring Function (Unchanged) ---
def calculate_triage_score(symptoms_text: str, weight_map: dict) -> tuple:
    total_score = 0
    recognized_symptoms = []
    search_string = symptoms_text.lower().replace(' ', '_')
    
    for symptom_phrase, weight in weight_map.items():
        if symptom_phrase in search_string:
            total_score += weight
            recognized_symptoms.append(f"{symptom_phrase} (Weight: {weight})")
            
    if total_score > triage_thresholds['MODERATE_MAX']:
        triage_level = "🔴 HIGH RISK (Score > 10)"
        triage_advice = "Seek **IMMEDIATE** professional medical attention."
    elif total_score > triage_thresholds['LOW_MAX']:
        triage_level = "🟡 MODERATE RISK (Score 6-10)"
        triage_advice = "Consult a doctor within 24-48 hours."
    else:
        triage_level = "🟢 LOW RISK (Score 1-5)"
        triage_advice = "Monitor symptoms and consider self-care/FAQ advice."
        
    return total_score, triage_level, triage_advice, recognized_symptoms


# --- 3. Multi-Graph Causal Visualization Function (NODE SIZING FIX) ---
def visualize_multi_causal_links(top_3_predictions, recognized_symptoms_list):
    """Generates three separate graphs, one for each of the top predictions."""
    
    print("\n--- CAUSAL GRAPH VISUALIZATION (Top 3) ---")
    
    symptom_names = [s.split('(')[0].strip() for s in recognized_symptoms_list if s]

    if not symptom_names:
        print("No high-weight symptoms were recognized to form a graph.")
        return

    num_predictions = len(top_3_predictions)
    
    # Set a common figure size based on the number of plots
    fig, axes = plt.subplots(1, num_predictions, figsize=(5 * num_predictions, 6))
    
    # Ensure axes is iterable even if only one subplot exists
    if num_predictions == 1:
        axes = [axes]

    for i, (disease, confidence) in enumerate(top_3_predictions):
        
        disease_node = disease.strip().lower().replace(' ', '_')
        
        # Build subgraph for this disease
        subgraph = nx.DiGraph() 
        for symptom in symptom_names:
            subgraph.add_edge(symptom, disease_node)

        # Visualization properties
        pos = nx.circular_layout(subgraph) 
        
        # INCREASE NODE SIZE and adjust font size for better fit
        NODE_SIZE_NEW = 4000 
        FONT_SIZE_NEW = 7.5
        
        node_colors = ['#FFC300' if node == disease_node else '#7D3C98' for node in subgraph.nodes]
        
        # Create cleaner labels for display (Title Case)
        node_labels = {node: node.replace('_', ' ').title() for node in subgraph.nodes}
        
        # Draw the graph on the current subplot
        nx.draw(subgraph, pos, 
                ax=axes[i], 
                with_labels=True, 
                labels=node_labels,
                node_size=NODE_SIZE_NEW, 
                node_color=node_colors, 
                font_size=FONT_SIZE_NEW, 
                font_color='white', 
                arrowstyle='-|>',
                arrowsize=15)
        
        axes[i].set_title(f"{disease.title()}\nConf: {confidence*100:.1f}%", fontsize=10)

    plt.tight_layout()
    plt.show(block=False) 
    plt.pause(0.1) 
    print("Multi-Graph Window initiated. Check your taskbar/desktop.")


# --- 4. Causal Scoreboard Function (Unchanged) ---
def explain_prediction_scoreboard(top_3_predictions, recognized_symptoms_list):
    """Provides quantified symptom contribution for all top 3 predictions."""
    
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


# --- 5. Main Chatbot Loop (Unchanged) ---
def run_chatbot(model, vectorizer, encoder, description_map):
    
    print("\n\n--- MEDILINGO AI CHATBOT (ENGLISH CORE) ---")
    print("Explainability Mode: Multi-Graph Visualization")
    print("Type 'exit' to end.")

    while True:
        user_input = input("\nEnter your symptoms: ").strip()
        
        if user_input.lower() == 'exit':
            break
        if not user_input:
            continue
            
        # A. Semantic Prediction
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.encode([cleaned_input]).astype(np.float64) 
        probabilities = model.predict_proba(input_vector)[0]
        
        # Get Top 3 Predictions
        top_k_indices = np.argsort(probabilities)[::-1][:3]
        top_3_predictions = []
        for i in top_k_indices:
            disease = encoder.inverse_transform([i])[0]
            confidence = probabilities[i]
            top_3_predictions.append((disease, confidence))
            
        primary_disease = top_3_predictions[0][0]

        # B. Severity Triage
        triage_score, triage_level, triage_advice, recognized_symptoms = calculate_triage_score(user_input, symptom_weight_map)
        
        
        print("\n=============================================")
        print(f"         INTEGRATED CHATBOT RESPONSE (Total Score: {triage_score})")
        print("=============================================")
        
        # 1. Prediction Output
        print("--- 1. DISEASE PREDICTION (Top 3) ---")
        
        for i, (disease, confidence) in enumerate(top_3_predictions):
            priority = "Primary" if i == 0 else f"Alternative {i}"
            print(f"  {priority}: **{disease}** ({confidence*100:.2f}%)")
            
            if i == 0:
                key = disease.strip().lower().replace(' ', '_')
                description = description_map.get(key, "Description not found in map.")
                print(f"  Description: {description}")


        # 2. Triage Output
        print("\n--- 2. SEVERITY TRIAGE ---")
        print(f"Risk Level: {triage_level}")
        print(f"Recognized Severity Tokens: {', '.join(recognized_symptoms) if recognized_symptoms else 'None'}")
        print(f"**Recommended Action:** {triage_advice}")
        
        # 3. Quantitative Scoreboard
        explain_prediction_scoreboard(top_3_predictions, recognized_symptoms)

        # 4. Causal Graph Visualization (Call the function)
        visualize_multi_causal_links(top_3_predictions, recognized_symptoms)

        print("=============================================")
        
# --- 6. Start the Chatbot ---
if __name__ == '__main__':
    run_chatbot(svm_semantic_model, vectorizer_model, le, disease_description_map)