import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Setup Paths ---
BASE_DIR = Path(__file__).parent
MEDIANS_FILE = BASE_DIR / "medians.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"

@st.cache_resource
def load_pipeline_components():
    # Load preprocessing artifacts
    medians = joblib.load(MEDIANS_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    # Langsung tembak file model.pkl, tanpa MLflow
    MODEL_FILE = BASE_DIR / "model.pkl"
    model = joblib.load(MODEL_FILE)
    
    return medians, scaler, model

st.set_page_config(page_title="Heart Attack Prediction", layout="centered")
st.title("🫀 Heart Attack Prediction App")
st.write("Aplikasi ini memprediksi kemungkinan serangan jantung berdasarkan data klinis menggunakan model LightGBM dari MLflow.")

# --- Load Artifacts & Model ---
@st.cache_resource
def load_pipeline_components():
    # Load preprocessing artifacts
    medians = joblib.load(MEDIANS_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    # Load model terbaru dari MLflow Experiment
    mlflow.set_tracking_uri("file://" + str(BASE_DIR / "mlruns")) 
    experiment = mlflow.get_experiment_by_name("Heart-Attack-Prediction")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    latest_run_id = runs.iloc[0].run_id
    model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/lightgbm-model")
    
    return medians, scaler, model

try:
    medians, scaler, model = load_pipeline_components()
    st.success("Model dan artifacts berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model/artifacts. Pastikan Anda sudah menjalankan pipeline.py. Error: {e}")
    st.stop()

# --- User Input Form ---
st.subheader("Masukkan Data Klinis Pasien")
features = medians.index.tolist()

# Membuat form input dinamis berdasarkan kolom fitur
user_input = {}
with st.form("prediction_form"):
    cols = st.columns(2)
    for i, feature in enumerate(features):
        # Membagi input ke dalam dua kolom agar UI lebih rapi
        with cols[i % 2]:
            user_input[feature] = st.number_input(f"{feature}", value=float(medians[feature]))
    
    submit_button = st.form_submit_button(label="Prediksi Risiko")

# --- Prediction Logic ---
if submit_button:
    # 1. Ubah input menjadi DataFrame
    input_df = pd.DataFrame([user_input])
    
    # 2. Isi missing values (kalau-kalau diperlukan)
    input_df.fillna(medians, inplace=True)
    
    # 3. Lakukan scaling
    input_scaled = scaler.transform(input_df)
    
    # 4. Prediksi dengan model
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    st.subheader("Hasil Prediksi")
    
    if prediction == 1:
        st.error(f"⚠️ **Tinggi Risiko Serangan Jantung** (Probabilitas: {probability:.2%})")
    else:
        st.success(f"✅ **Rendah Risiko Serangan Jantung** (Probabilitas: {probability:.2%})")