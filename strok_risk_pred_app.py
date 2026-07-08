import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Page Configuration (Light & Professional Theme)
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS for clean UI styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2b7a78;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #17252a; color: white; }
    </style>
""", unsafe_allow_html=True)

# 2. Load the Saved Model Safely
@st.cache_resource
def load_model():
    return joblib.load("model_dt.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'model_dt.pkl' not found. Please save your model first.")
    st.stop()

# 3. App Header
st.title("🩺 Stroke Risk Assessment System")
st.markdown("Provide patient clinical metrics below to evaluate stroke vulnerability using our optimized Decision Tree model.")
st.write("---")

# 4. Organizing Inputs into Columns (Attractive UI Layout)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=0, max_value=120, value=45, step=1)
    hypertension = st.selectbox("Hypertension (High BP)", options=["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease History", options=["No", "Yes"])

with col2:
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=105.0, step=1.0)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=28.0, step=0.1)

# 5. Data Transformation (Converting UI text to binary matching your training)
hyp_val = 1 if hypertension == "Yes" else 0
heart_val = 1 if heart_disease == "Yes" else 0

# 6. Prediction Logic & Clean Output Display
st.write("")
if st.button("Analyze Stroke Risk"):
    # Creating a 2D Array / DataFrame to match training shape
    input_data = np.array([[age, hyp_val, heart_val, avg_glucose_level, bmi]])
    
    # Predict
    prediction = model.predict(input_data)
    
    st.write("---")
    if prediction[0] == 1:
        st.error("### ⚠️ High Risk Warning\nThe system predicts a significant risk of stroke based on the clinical parameters. Immediate medical consultation is highly recommended.")
    else:
        st.success("### ✅ Low Risk Confirmed\nThe system predicts a low risk of stroke. Maintain a healthy lifestyle and regular checkups.")