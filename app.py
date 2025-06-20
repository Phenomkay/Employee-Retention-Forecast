import streamlit as st
import joblib
import numpy as np
import base64

# === STYLE ===
def set_bg_and_fonts(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=League+Spartan&family=Poppins&display=swap');

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            font-family: 'Poppins', sans-serif;
            color: navy;
        }}

        h1, h2, h3, h4, p, label, .css-1cpxqw2 {{
            font-family: 'League Spartan', sans-serif !important;
            color: navy !important;
        }}

        .stTextInput > div > div > input {{
            background-color: navy !important;
            color: white !important;
        }}

        .stNumberInput input {{
            background-color: navy !important;
            color: white !important;
        }}

        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: navy !important;
            color: white !important;
        }}

        .stSlider > div {{
            color: navy !important;
        }}

        .stButton > button {{
            background-color: white !important;
            color: navy !important;
        }}

        .stAlert {{
            color: navy !important;
        }}

        footer {{
            color: navy !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# === BACKGROUND ===
set_bg_and_fonts("working_employee_2.png")

# === TITLE ===
st.title("Employee Attrition Prediction App")
st.subheader("Will your employee stay or leave?")

# === Load Model & Features ===
model = joblib.load("employee_retention_model.joblib")
feature_list = joblib.load("model_features.joblib")

# === Input Fields ===
st.markdown("### Enter Employee Information")
satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01)
last_evaluation = st.number_input("Last Evaluation", min_value=0.0, max_value=1.0, step=0.01)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350)
time_spend_company = st.number_input("Years at Company", min_value=0, max_value=10)
Work_accident = st.selectbox("Had Work Accident?", [0, 1])
promotion_last_5years = st.selectbox("Promoted in Last 5 Years?", [0, 1])
salary = st.selectbox("Salary Level", ["low", "medium", "high"])
Department = st.selectbox("Department", [
    "sales", "accounting", "hr", "technical", "support",
    "management", "IT", "product_mng", "marketing", "RandD"
])

# === Encode Inputs ===
salary_map = {"low": 0, "medium": 1, "high": 2}
departments = [
    'Department_RandD', 'Department_accounting', 'Department_hr',
    'Department_management', 'Department_marketing', 'Department_product_mng',
    'Department_sales', 'Department_support', 'Department_technical'
]

# Base input array
input_data = [
    satisfaction_level, last_evaluation, number_project,
    average_montly_hours, time_spend_company,
    Work_accident, promotion_last_5years, salary_map[salary]
]

# One-hot encode Department
for dept in departments:
    input_data.append(1 if dept == f'Department_{Department}' else 0)

# === Prediction ===
if st.button("Predict Attrition"):
    prediction = model.predict([input_data])[0]
    if prediction == 1:
        st.error("⚠️ This employee is likely to leave the company.")
    else:
        st.success("✅ This employee is likely to stay.")