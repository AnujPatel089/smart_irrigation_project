import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load('irrigation_model.pkl')

model = load_model()

st.title("🪴 Smart Irrigation Predictor")
st.write("ML predictions using your Smart_irrigation_dataset.csv")

# === YOUR 6 EXACT FEATURES ===
temp = st.slider("temperature_C (°C)", 10.0, 40.0, 25.0)
hum = st.slider("humidity_% (%)", 20.0, 100.0, 70.0)
soil = st.slider("soil_moisture_% (%)", 0.0, 50.0, 25.0)
rain = st.slider("rainfall_mm", 0.0, 20.0, 2.0)
solar = st.slider("solar_radiation_MJ_m2_day", 0.0, 30.0, 15.0)
wind = st.slider("wind_speed_m_s", 0.0, 10.0, 2.0)

if st.button("🚀 Predict Irrigation Need", type="primary"):
    # EXACT 6-FEATURE ORDER from notebook
    input_data = np.array([[temp, hum, soil, rain, solar, wind]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    
    if pred == 1:
        st.success("🚿 **IRRIGATE NOW**")
    else:
        st.info("✅ **No Irrigation Needed**")
    
    st.metric("Irrigate Probability", f"{prob[1]:.1%}")
    st.bar_chart({"Irrigate": prob[1], "No": prob[0]})

st.sidebar.success("✅ Model: RandomForest\n📊 Features: 6\n🎯 Dataset: Yours")