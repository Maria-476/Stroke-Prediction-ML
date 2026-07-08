import streamlit as st
import joblib
import numpy as np
import time

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"   # FIX 2: sidebar toggle visible
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("model_dt.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file not found!")
    st.stop()

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp {
    background: linear-gradient(135deg, #F6FCFF, #EAF8FF);
}

/* FIX 1: Input labels visible */
label, div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #1f2937 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Header */
.main-header {
    background: linear-gradient(90deg, #0EA5E9, #2563EB);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}
.main-header h1 { color: white; font-size: 40px; margin-bottom: 5px; }
.main-header p  { color: white; font-size: 18px; margin: 0; }

/* Section titles */
.section-title {
    font-size: 20px;
    font-weight: bold;
    color: #0F766E;
    margin-bottom: 12px;
    padding: 10px 0;
    border-bottom: 2px solid #0EA5E9;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #14B8A6, #0EA5E9);
    color: white;
    font-size: 20px;
    font-weight: bold;
    padding: 14px;
    border: none;
    border-radius: 12px;
    width: 100%;
    transition: 0.3s;
}
.stButton > button:hover { transform: scale(1.02); }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0F172A; }
section[data-testid="stSidebar"] * { color: white; }

/* FIX 4 & 6: Result cards — dark readable text */
.success-card {
    background: #DCFCE7;
    padding: 25px;
    border-radius: 15px;
    border-left: 8px solid #22C55E;
    margin-bottom: 15px;
}
.success-card h2, .success-card h3,
.success-card p,  .success-card li { color: #14532d; }
.success-card h2 { font-size: 26px; margin-bottom: 10px; }
.success-card h3 { font-size: 20px; margin-top: 18px; margin-bottom: 8px; }
.success-card p, .success-card li { font-size: 16px; line-height: 2; }
.success-card ul { padding-left: 20px; }

.danger-card {
    background: #FEE2E2;
    padding: 25px;
    border-radius: 15px;
    border-left: 8px solid #EF4444;
    margin-bottom: 15px;
}
.danger-card h2, .danger-card h3,
.danger-card p,  .danger-card li { color: #7f1d1d; }
.danger-card h2 { font-size: 26px; margin-bottom: 10px; }
.danger-card h3 { font-size: 20px; margin-top: 18px; margin-bottom: 8px; }
.danger-card p, .danger-card li { font-size: 16px; line-height: 2; }
.danger-card ul { padding-left: 20px; }

.info-card {
    background: #DBEAFE;
    padding: 20px;
    border-radius: 12px;
}
.info-card h3, .info-card p { color: #1e3a5f; }
.info-card h3 { font-size: 20px; margin-bottom: 10px; }
.info-card p  { font-size: 15px; line-height: 1.8; }

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="main-header">
    <h1>🧠 Stroke Risk Prediction System</h1>
    <p>Predict Stroke Risk using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.title("🏥 About")
    st.info("""
### Stroke Risk Predictor
This application predicts whether a patient
is at risk of Stroke using a trained
Decision Tree Machine Learning model.
""")
    st.divider()
    st.markdown("### 📊 Model Info")
    st.write("**Algorithm:** Decision Tree")
    st.write("**Recall Score:** 0.88")
    st.write("**Version:** 1.0")
    st.write("**Developer:** Maria Anwar")
    st.write("**Purpose:** Educational")
    st.divider()
    st.success("💙 Stay Healthy")
    st.caption("""
✔ Exercise Regularly
✔ Maintain Healthy Weight
✔ Control Blood Pressure
✔ Stop Smoking
✔ Eat Healthy Food
""")

# ==========================================================
# INPUT SECTIONS
# ==========================================================

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">👤 Personal Information</div>',
                unsafe_allow_html=True)
    gender     = st.selectbox("Gender", ["Male", "Female", "Other"])
    age        = st.number_input("Age", min_value=1, max_value=100, value=45)
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    residence  = st.selectbox("Residence Type", ["Urban", "Rural"])
    work_type  = st.selectbox("Work Type",
                    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

with right:
    st.markdown('<div class="section-title">❤️ Health Information</div>',
                unsafe_allow_html=True)
    hypertension  = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    avg_glucose   = st.number_input("Average Glucose Level",
                        min_value=50.0, max_value=300.0, value=100.0)
    bmi           = st.number_input("Body Mass Index (BMI)",
                        min_value=10.0, max_value=60.0, value=25.0)
    smoking       = st.selectbox("Smoking Status",
                        ["never smoked", "formerly smoked", "smokes", "Unknown"])

# ==========================================================
# ENCODING MAPS
# ==========================================================

gender_map    = {"Female": 0, "Male": 1, "Other": 2}
married_map   = {"No": 0, "Yes": 1}
work_map      = {"Govt_job": 0, "Never_worked": 1, "Private": 2,
                 "Self-employed": 3, "children": 4}
residence_map = {"Rural": 0, "Urban": 1}
smoking_map   = {"Unknown": 0, "formerly smoked": 1,
                 "never smoked": 2, "smokes": 3}
binary_map    = {"No": 0, "Yes": 1}

# ==========================================================
# PREDICT BUTTON  (FIX 3: single br, no st.write gaps)
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🩺 Analyze Stroke Risk")

# ==========================================================
# PREDICTION LOGIC
# ==========================================================

if predict:

    with st.spinner("Analyzing patient data..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    input_data = np.array([[
        gender_map[gender],
        age,
        binary_map[hypertension],
        binary_map[heart_disease],
        married_map[ever_married],
        work_map[work_type],
        residence_map[residence],
        avg_glucose,
        bmi,
        smoking_map[smoking]
    ]])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.divider()

    # PATIENT SUMMARY
    st.subheader("📋 Patient Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Age",     f"{age} yrs")
    with c2: st.metric("BMI",     f"{bmi:.1f}")
    with c3: st.metric("Glucose", f"{avg_glucose:.1f}")
    with c4: st.metric("BP",      hypertension)

    # PROBABILITY BAR  (FIX 1: /100)
    st.subheader("📊 Estimated Stroke Risk")
    st.progress(min(int(probability), 100) / 100)
    color = "🟢" if probability < 30 else ("🟡" if probability < 70 else "🔴")
    st.markdown(f"### {color} Risk Probability: **{probability:.1f}%**")

    # RESULT CARDS  (FIX 4,5,6: all content inside one div, no floating emojis)
    if prediction == 1:
        st.markdown("""
        <div class="danger-card">
            <h2>⚠️ High Stroke Risk Detected</h2>
            <p>The model predicts this patient has a higher probability of stroke.
               Please consult a qualified healthcare professional immediately.
               This prediction should NOT replace medical diagnosis.</p>
            <h3>💡 Health Recommendations</h3>
            <ul>
                <li>Monitor Blood Pressure Regularly</li>
                <li>Keep Blood Sugar Under Control</li>
                <li>Quit Smoking Immediately</li>
                <li>Exercise Daily (30 mins minimum)</li>
                <li>Reduce Salt Intake</li>
                <li>Eat More Fruits and Vegetables</li>
                <li>Maintain Healthy Body Weight</li>
                <li>Follow Your Doctor's Advice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-card">
            <h2>✅ Low Stroke Risk</h2>
            <p>The model predicts a lower risk of stroke.
               Keep maintaining a healthy lifestyle.
               Regular medical checkups are always recommended.</p>
            <h3>🌿 Healthy Lifestyle Tips</h3>
            <ul>
                <li>Stay Physically Active</li>
                <li>Eat Healthy Food</li>
                <li>Drink Plenty of Water</li>
                <li>Sleep 7–8 Hours Daily</li>
                <li>Maintain Healthy Weight</li>
                <li>Avoid Smoking</li>
                <li>Reduce Stress</li>
                <li>Monitor Blood Pressure Regularly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # DISCLAIMER
    st.divider()
    st.markdown("""
    <div class="info-card">
        <h3>⚕️ Disclaimer</h3>
        <p>This application is developed for <strong>educational purposes only</strong>.
        Predictions generated by this Machine Learning model should <strong>not</strong>
        be considered medical advice, diagnosis, or treatment.
        Always consult a licensed healthcare professional for proper medical evaluation.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Developed with ❤️ using Streamlit | Stroke Risk Prediction System | © 2026")