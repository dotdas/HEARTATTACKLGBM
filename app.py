import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Setup Paths ---
BASE_DIR = Path(__file__).parent
MEDIANS_FILE = BASE_DIR / "medians.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"
MODEL_FILE = BASE_DIR / "model.pkl" 

st.set_page_config(page_title="Heart Attack Prediction", layout="centered")
st.title("🫀 Heart Attack Prediction App")
st.write("Aplikasi ini memprediksi kemungkinan serangan jantung berdasarkan data klinis menggunakan model LightGBM.")

# --- Load Artifacts & Model (Murni pakai Joblib, tanpa MLflow) ---
@st.cache_resource
def load_pipeline_components():
    medians = joblib.load(MEDIANS_FILE)
    scaler = joblib.load(SCALER_FILE)
    model = joblib.load(MODEL_FILE)
    
    return medians, scaler, model

try:
    medians, scaler, model = load_pipeline_components()
    st.success("Model dan artifacts berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat file .pkl. Pastikan medians.pkl, scaler.pkl, dan model.pkl ada di GitHub. Error: {e}")
    st.stop()

# --- User Input Form ---
st.subheader("Masukkan Data Klinis Pasien")
features = medians.index.tolist()

user_input = {}
with st.form("prediction_form"):
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            user_input[feature] = st.number_input(f"{feature}", value=float(medians[feature]))
    
    submit_button = st.form_submit_button(label="Prediksi Risiko")

# --- Prediction Logic ---
if submit_button:
    input_df = pd.DataFrame([user_input])
    input_df.fillna(medians, inplace=True)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    st.subheader("Hasil Prediksi")
    
    if prediction == 1:
        st.error(f"⚠️ **Tinggi Risiko Serangan Jantung** (Probabilitas: {probability:.2%})")
    else:
        st.success(f"✅ **Rendah Risiko Serangan Jantung** (Probabilitas: {probability:.2%})")
