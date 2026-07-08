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
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("model_dt.pkl")     # Change filename if needed

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

.stApp{
    background:linear-gradient(135deg,#F6FCFF,#EAF8FF);
}

/* Header */

.main-header{
background:linear-gradient(90deg,#0EA5E9,#2563EB);
padding:25px;
border-radius:18px;
text-align:center;
box-shadow:0px 8px 20px rgba(0,0,0,.15);
margin-bottom:20px;
}

.main-header h1{
color:white;
font-size:40px;
margin-bottom:5px;
}

.main-header p{
color:white;
font-size:18px;
}

/* Cards */

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 5px 18px rgba(0,0,0,.12);
margin-bottom:18px;
}

/* Section title */

.section-title{
font-size:22px;
font-weight:bold;
color:#0F766E;
margin-bottom:10px;
}

/* Button */

.stButton>button{
background:linear-gradient(90deg,#14B8A6,#0EA5E9);
color:white;
font-size:20px;
font-weight:bold;
padding:14px;
border:none;
border-radius:12px;
width:100%;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.02);
}

/* Sidebar */

section[data-testid="stSidebar"]{
background:#0F172A;
}

section[data-testid="stSidebar"] *{
color:white;
}

/* Result Cards */

.success-card{
background:#DCFCE7;
padding:20px;
border-radius:15px;
border-left:8px solid #22C55E;
}

.danger-card{
background:#FEE2E2;
padding:20px;
border-radius:15px;
border-left:8px solid #EF4444;
}

.info-card{
background:#DBEAFE;
padding:15px;
border-radius:12px;
}

.metric-box{
background:white;
padding:15px;
border-radius:12px;
text-align:center;
box-shadow:0px 4px 12px rgba(0,0,0,.12);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="main-header">
<h1>🧠 Stroke Risk Prediction System</h1>

<p>
Predict Stroke Risk using Machine Learning
</p>

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

    st.markdown("### 📊 Model")

    st.write("**Algorithm:** Decision Tree")

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

left,right = st.columns(2)

# ----------------------------------------------------------
# LEFT SIDE
# ----------------------------------------------------------

with left:

    st.markdown("""
<div class="card">

<div class="section-title">

👤 Personal Information

</div>

</div>
""", unsafe_allow_html=True)

    gender = st.selectbox(
        "Gender",
        ["Male","Female","Other"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=45
    )

    ever_married = st.selectbox(
        "Ever Married",
        ["Yes","No"]
    )

    residence = st.selectbox(
        "Residence Type",
        ["Urban","Rural"]
    )

    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )

# ----------------------------------------------------------
# RIGHT SIDE
# ----------------------------------------------------------

with right:

    st.markdown("""
<div class="card">

<div class="section-title">

❤️ Health Information

</div>

</div>
""", unsafe_allow_html=True)

    hypertension = st.selectbox(
        "Hypertension",
        ["No","Yes"]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        ["No","Yes"]
    )

    avg_glucose = st.number_input(
        "Average Glucose Level",
        min_value=50.0,
        max_value=300.0,
        value=100.0
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    smoking = st.selectbox(
        "Smoking Status",
        [
            "never smoked",
            "formerly smoked",
            "smokes",
            "Unknown"
        ]
    )

# ==========================================================
# ENCODING DICTIONARIES
# ==========================================================

gender_map = {
    "Female":0,
    "Male":1,
    "Other":2
}

married_map = {
    "No":0,
    "Yes":1
}

work_map = {
    "Govt_job":0,
    "Never_worked":1,
    "Private":2,
    "Self-employed":3,
    "children":4
}

residence_map = {
    "Rural":0,
    "Urban":1
}

smoking_map = {
    "Unknown":0,
    "formerly smoked":1,
    "never smoked":2,
    "smokes":3
}

binary_map = {
    "No":0,
    "Yes":1
}

st.write("")

predict = st.button("🩺 Analyze Stroke Risk")

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    # Small loading animation
    with st.spinner("Analyzing patient data..."):
        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    # Prepare input in SAME ORDER as training
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

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability (if available)
    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1] * 100

    st.write("")
    st.divider()

    # =====================================================
    # PATIENT SUMMARY
    # =====================================================

    st.subheader("📋 Patient Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Age", f"{age} yrs")

    with c2:
        st.metric("BMI", f"{bmi:.1f}")

    with c3:
        st.metric("Glucose", f"{avg_glucose:.1f}")

    with c4:
        st.metric("BP", hypertension)

    st.write("")

    # =====================================================
    # PROBABILITY
    # =====================================================

    if probability is not None:

        st.subheader("📊 Estimated Stroke Risk")

        st.progress(min(int(probability),100)/100)

        if probability < 30:
            color = "🟢"

        elif probability < 70:
            color = "🟡"

        else:
            color = "🔴"

        st.markdown(
            f"### {color} Estimated Risk Probability : **{probability:.1f}%**"
        )

    st.write("")

    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        st.markdown("""
        <div class="danger-card">

        <h2>
        ⚠️ High Stroke Risk Detected
        </h2>

        <p style="font-size:18px;">

        The model predicts that this patient has
        a higher probability of stroke.

        </p>

        </div>
        """,
        unsafe_allow_html=True)

        st.error("""
Please consult a qualified healthcare professional immediately.

This prediction should NOT replace medical diagnosis.
""")

        st.subheader("💡 Health Recommendations")

        st.warning("""
• Monitor Blood Pressure Regularly

• Keep Blood Sugar Under Control

• Quit Smoking

• Exercise Daily

• Reduce Salt Intake

• Eat More Fruits & Vegetables

• Maintain Healthy Body Weight

• Follow Your Doctor's Advice
""")

    else:

        st.markdown("""
        <div class="success-card">

        <h2>

        ✅ Low Stroke Risk

        </h2>

        <p style="font-size:18px;">

        The model predicts a lower risk of stroke.

        </p>

        </div>
        """,
        unsafe_allow_html=True)

        st.success("""
Keep maintaining a healthy lifestyle.

Regular medical checkups are always recommended.
""")

        st.subheader("🌿 Healthy Lifestyle Tips")

        st.info("""
• Stay Physically Active

• Eat Healthy Food

• Drink Plenty of Water

• Sleep 7–8 Hours

• Maintain Healthy Weight

• Avoid Smoking

• Reduce Stress

• Monitor Blood Pressure
""")

    st.divider()

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.markdown("""
<div class="info-card">

### ⚕️ Disclaimer

This application is developed for **educational purposes only**.

Predictions generated by this Machine Learning model should **not**
be considered medical advice, diagnosis, or treatment.

Always consult a licensed healthcare professional for proper medical
evaluation.

</div>
""",
unsafe_allow_html=True)

# ==========================================================
# FOOTER
# ==========================================================

st.write("")
st.write("")

st.caption(
    "Developed with ❤️ using Streamlit | Stroke Risk Prediction System | © 2026"
)