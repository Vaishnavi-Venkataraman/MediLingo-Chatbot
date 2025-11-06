# 🩺 MediLingo AI Health Assistant: A Multilingual System for Symptom Diagnosis and Triage

**Authors:**  
- 23PT11 – Harshil Bhavik Momaya  
- 23PT37 – Vaishnavi V  

---

## 🎯 Features
- **MediLingo AI** is a robust multilingual health assistant designed to provide instant, context-aware diagnosis and emergency triage.  
- The system interprets **multilingual and conversational inputs** (Hindi, Tamil, English) using a **Fine-Tuned Multilingual Sentence-BERT (SBERT)** model paired with a high-accuracy **SVM classifier**.  
- It ensures **context retention** through cumulative symptom memory and supports **real-time risk triage** for safe, explainable diagnostic outcomes.  
- The assistant integrates **Causal Visualization** and **Emergency Hospital Links**, ensuring both interpretability and actionability for users in medical need.

---

## 📊 Datasets
MediLingo AI combines structured and unstructured datasets for training and evaluation:

| Dataset Type | Description | Source |
|---------------|-------------|---------|
| **Semantic Prediction Data** | Natural-language symptom-to-disease mappings (e.g., `Symptom2Disease.csv`) | Kaggle |
| **Structured Severity Data** | Binary symptom flags and numerical severity weights (`Training.csv`, `Symptom-severity.csv`) | Kaggle |
| **Description Data** | Custom dataset linking each disease to descriptive text (`Symptom_Description.csv`) | Kaggle |

---

## ⚙️ Methods and Implementation
### 🧩 Conversational Memory and Context
- Maintains **cumulative symptom memory**, ensuring all user-entered symptoms (e.g., “headache and also throat pain”) are analyzed together until a **‘New Chat’** is initiated.  
- Supports **multi-turn conversations**, allowing continuous updates to the symptom list.  
- Stores and reuses the **raw English disease name** for contextual follow-up queries, ensuring reliability across all supported languages.

### 🌐 Multilingual Transition Bridge
- Implements multilingual support using **mBART-Large-50 (Multilingual BART)**.  
- Acts as a **sequence-to-sequence (Seq2Seq)** translation bridge:
          User Input (Hindi/Tamil) → Translate to English → Process via SBERT → Translate Output Back → Display (Hindi/Tamil)
- Ensures accurate medical comprehension even with noisy, regional, or mixed-language input.

### 🚨 Risk Assessment and Triage System
- Uses **Rule-Based Severity Scoring (1–7)** to assess symptom seriousness.  
- Prevents alias symptom duplication for accurate risk calculation.  
- Integrates **clinical override logic**:
- If model confidence < 35% and symptoms contain high-risk keywords (e.g., `urinary_pain`), override to a safe, specific diagnosis (e.g., `Urinary Tract Infection`).

---

## 📈 Model Evaluation, Results & Visualization
- **Accuracy:** 97.5% (evaluated on multilingual symptom datasets).  
- **Causal Graph Visualization:**  
- Dynamically generated **Base64 causal graph images** show symptom–disease relationships.  
- Displayed inline within the chat for interpretability.  
- **Emergency Guidance:**  
- For critical triage results, generates a **clickable Google Maps link** for the nearest hospital — no external API dependency.  
- **UI/UX:**  
- Clean, professional **multi-turn chat interface** optimized for both mobile and desktop.  
- Uses a **FastAPI backend** with secure session management and dynamic chat updates.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language Models** | Multilingual SBERT, mBART-Large-50 |
| **Classifier** | SVM |
| **Backend Framework** | FastAPI |
| **Frontend** | HTML, CSS, JavaScript |
| **Visualization** | Matplotlib (Causal Graphs) |
| **Deployment** | Local/Cloud (via Uvicorn server) |

---

## 🚀 How to Run

### Prerequisites:
- 🐍 **Python 3.10+**
- ⚙️ **FastAPI & Uvicorn**
- 🧠 **Transformers**, **Sentence-Transformers**, **Scikit-learn**
- 🧾 Install dependencies:
```bash
pip install -r requirements.txt
````
### Setup Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MediLingo-Chatbot.git
   cd MediLingo-Chatbot
   ```
2. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```
3. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```
4. Interact with MediLingo AI through the integrated **chat interface**.
