import streamlit as st
st.set_page_config(page_title="Healthchat Assistant", page_icon="🩺", layout="centered")

import pandas as pd
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import string
import difflib
import speech_recognition as sr
from fpdf import FPDF
from datetime import datetime
import json
import os
import PyPDF2

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

tokenizer = TreebankWordTokenizer()

@st.cache_data
def load_disease_data():
    return pd.read_csv("final_healthcare_disease_dataset.csv")

disease_df = load_disease_data()

@st.cache_data
def extract_all_symptoms():
    all_symptoms = set()
    for symptoms in disease_df["Symptoms"]:
        all_symptoms.update([sym.strip().lower() for sym in symptoms.split(",")])
    return all_symptoms

known_symptoms = extract_all_symptoms()
@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad"
    )

qa_model = load_qa_model()

pdf_context = ""

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def autocorrect_symptoms(tokens):
    corrected = []
    for token in tokens:
        if token in known_symptoms:
            corrected.append(token)
        else:
            matches = difflib.get_close_matches(token, known_symptoms, n=1, cutoff=0.8)
            corrected.append(matches[0] if matches else token)
    return corrected

def preprocess_input(text):
    tokens = tokenizer.tokenize(text.lower())
    STOP_WORDS = set(stopwords.words("english"))
    tokens = [
        word for word in tokens
        if word not in STOP_WORDS and word not in string.punctuation
    ]
    return autocorrect_symptoms(tokens)

def predict_disease(user_symptoms):
    max_match = 0
    predicted = None
    for _, row in disease_df.iterrows():
        disease_symptoms = [sym.strip().lower() for sym in row["Symptoms"].split(",")]
        match_count = len(set(user_symptoms) & set(disease_symptoms))
        if match_count > max_match:
            max_match = match_count
            predicted = row
    return predicted

def healthcare_chatbot(user_input, context):
    tokens = preprocess_input(user_input)
    result = predict_disease(tokens)
    if result is not None:
        return (
            f"<div class='response'>"
            f"<h4>🦠 Predicted Disease: <code>{result['Disease']}</code></h4>"
            f"<p><b>Description:</b> {result['Description']}</p>"
            f"<p><b>Prevention:</b> {result['Prevention']}</p>"
            f"<p style='color:#555;'>💡 Consult a doctor for professional diagnosis.</p>"
            f"</div>", result
        )
    else:
        try:
            answer = qa_model(question=user_input, context=context)["answer"]
            return f"<p class='response'>🧠 <b>Medical Info:</b> {answer}</p>", None
        except Exception:
            return "<p style='color:red;'>⚠️ Sorry, I'm unable to process that right now.</p>", None

def generate_pdf_report(symptoms, severity, diagnosis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Healthchat Assistant Report", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Symptoms Entered: {symptoms}")
    pdf.multi_cell(0, 10, f"Symptom Severity (1-10): {severity}")
    if diagnosis is not None:
        pdf.multi_cell(0, 10, f"Predicted Disease: {diagnosis['Disease']}")
        pdf.multi_cell(0, 10, f"Description: {diagnosis['Description']}")
        pdf.multi_cell(0, 10, f"Prevention: {diagnosis['Prevention']}")
    else:
        pdf.multi_cell(0, 10, "No clear disease detected.")
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

def sidebar_medicine_routine():
    st.sidebar.header("💊 Medicine Routine")
    if "med_schedule" not in st.session_state:
        if os.path.exists("med_schedule.json"):
            with open("med_schedule.json", "r") as f:
                st.session_state.med_schedule = json.load(f)
        else:
            st.session_state.med_schedule = []

    with st.sidebar.form("med_form"):
        med_name = st.text_input("Medicine Name")
        dose_time = st.time_input("Time to Take")
        if st.form_submit_button("Add to Routine"):
            st.session_state.med_schedule.append({"medicine": med_name, "time": dose_time.strftime("%H:%M")})
            with open("med_schedule.json", "w") as f:
                json.dump(st.session_state.med_schedule, f)
            st.success(f"Added {med_name} at {dose_time.strftime('%H:%M')}")

    if st.session_state.med_schedule:
        st.sidebar.markdown("### 📌 Your Medicines")
        for med in st.session_state.med_schedule:
            st.sidebar.markdown(f"- {med['medicine']} at {med['time']}")

    if st.sidebar.button("🔄 Refresh Reminders"):
        st.rerun()

def sidebar_diet_plans():
    with st.sidebar:
        st.markdown("---")
        st.header("🍽️ Basic Diet Plans")
        st.markdown("**Children (5-12 yrs)**")
        st.markdown("- 🥛 Milk & Dairy\n- 🥦 Veggies & Fruits\n- 🥪 Whole Grains\n- 🥩 Protein (Eggs, Lentils)")

        st.markdown("**Young Adults (18-30 yrs)**")
        st.markdown("- 🥗 Salads & Fruits\n- 🍗 Lean Protein\n- 🍚 Brown Rice, Oats\n- 💧 Stay Hydrated")

        st.markdown("**Adults (30-60 yrs)**")
        st.markdown("- 🥬 Fiber-rich food\n- 🐟 Omega-3s\n- 🥛 Low-fat Dairy\n- 🚫 Limit Sugar & Salt")

        st.markdown("**Elderly (60+ yrs)**")
        st.markdown("- 🥣 Soft, Nutrient-dense\n- 🧀 Calcium & Vitamin D\n- 🥕 Easy-to-digest Veggies\n- 💦 Hydration is key")

        st.markdown("**Women (All ages)**")
        st.markdown("- 🫘 Iron-rich food\n- 🥜 Folic acid\n- 🥑 Healthy Fats\n- 🥤 Lots of Water")

        st.markdown("**Men (All ages)**")
        st.markdown("- 🥩 Protein-packed\n- 🍅 Lycopene (Tomatoes)\n- 🍞 Whole Grains\n- 🥗 Balanced Meals")

def check_medicine_reminders():
    now = datetime.now().strftime("%H:%M")
    now_time = datetime.strptime(now, "%H:%M")
    due_meds = []

    for med in st.session_state.med_schedule:
        med_time = datetime.strptime(med["time"], "%H:%M")
        if abs((now_time - med_time).total_seconds()) <= 300:
            due_meds.append(med)
    return due_meds

def main():
    global pdf_context
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        background-color: #f0f4f8;
        color: #222;
    }
    .main-title {
        text-align: center;
        font-size: 2.6em;
        background: linear-gradient(to right, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px #4facfe; }
        to { text-shadow: 0 0 20px #00f2fe; }
    }
    .response {
        background: #f4f9fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4facfe;
        margin-top: 1rem;
        color: #111;
    }
    </style>
    """, unsafe_allow_html=True)

    sidebar_medicine_routine()
    sidebar_diet_plans()

    reminders = check_medicine_reminders()
    if reminders:
        for med in reminders:
            st.toast(f"⏰ Time to take {med['medicine']}!", icon="💊")

    st.markdown("<h1 class='main-title'>🩺 Healthchat Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Your futuristic AI-powered health companion 👾</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📎 Upload a PDF for medical content (optional)", type=["pdf"])
    if uploaded_file:
        pdf_context = read_pdf(uploaded_file)
        st.success("✅ PDF content loaded for QA model.")

    with st.expander("🎧 Or use voice input"):
        if st.button("Start Listening"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("🎧 Listening... Speak your symptoms.")
                try:
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio)
                    st.success(f"You said: {text}")
                    st.session_state.voice_input = text
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    user_input = st.text_input("💬 What symptoms are you experiencing?", value=st.session_state.get("voice_input", ""))
    severity = st.slider("🔢 Symptom severity (1: Mild – 10: Severe)", 1, 10, 5)

    if st.button("🔍 Analyze Symptoms"):
        if user_input.strip():
            st.markdown(f"<div class='response'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)
            response, diagnosis = healthcare_chatbot(user_input, pdf_context)
            st.markdown(response, unsafe_allow_html=True)
            pdf_data = generate_pdf_report(user_input, severity, diagnosis)
            st.download_button("📄 Download PDF Report", data=pdf_data, file_name="health_report.pdf")
        else:
            st.warning("⚠️ Please enter symptoms to continue.")

    st.markdown("<hr style='margin-top:2rem; border-color:#ccc;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>⚠️ This assistant does not replace professional medical advice.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
