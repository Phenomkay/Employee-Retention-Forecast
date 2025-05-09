import streamlit as st
import pandas as pd
import joblib
import base64

# === Load model, scaler, and feature columns ===
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")

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

set_bg_and_fonts("working_employee_2.png")

# === TITLE ===
st.title("Employee Attrition Prediction App")
st.subheader("Will your employee stay or leave?")

# === INPUTS ===
satisfaction_level = st.number_input("Satisfaction Level (0.0 - 1.0)", 0.0, 1.0, 0.5)
last_evaluation = st.number_input("Last Evaluation Score (0.0 - 1.0)", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, value=160)
time_spend_company = st.number_input("Years at Company", min_value=1, max_value=10, value=3)
work_accident = st.selectbox("Work Accident", ["No", "Yes"])
promotion_last_5years = st.selectbox("Promoted in Last 5 Years", ["No", "Yes"])
department = st.selectbox("Department", [
    'sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
    'RandD', 'accounting', 'hr', 'management'
])
salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# === PREDICT BUTTON ===
if st.button("Predict", key="predict_button"):
    # Convert inputs
    Work_accident = 1 if work_accident == "Yes" else 0
    promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

    input_data = pd.DataFrame({
        'satisfaction_level': [satisfaction_level],
        'last_evaluation': [last_evaluation],
        'number_project': [number_project],
        'average_montly_hours': [average_montly_hours],
        'time_spend_company': [time_spend_company],
        'Work_accident': [Work_accident],
        'promotion_last_5years': [promotion_last_5years],
        'Department': [department],
        'salary': [salary]
    })

    # === Feature Engineering ===
    input_data['is_very_low_project_count'] = (input_data['number_project'] == 2).astype(int)
    input_data['is_high_project_count'] = (input_data['number_project'] > 5).astype(int)
    input_data['very_high_evaluation'] = (input_data['last_evaluation'] > 0.9).astype(int)
    input_data['low_salary_high_hours'] = ((input_data['salary'] == 'low') & (input_data['average_montly_hours'] > 250)).astype(int)
    input_data['low_satisfaction_no_promo'] = ((input_data['satisfaction_level'] < 0.4) & (input_data['promotion_last_5years'] == 0)).astype(int)
    input_data['workload_index'] = input_data['number_project'] * input_data['average_montly_hours']
    input_data['within_first_3_years'] = (input_data['time_spend_company'] <= 3).astype(int)

    # === Mapping categorical features ===
    input_data['salary'] = input_data['salary'].map({'low': 0, 'medium': 1, 'high': 2})
    input_data['Department'] = input_data['Department'].map({'sales': 0, 'accounting': 1, 'hr': 2, 'technical': 3, 'support': 4, 'management': 5, 'IT': 6, 'product_mng': 7, 'marketing': 8, 'RandD': 9})

    # === Ensure Column Order Before Scaling ===
    input_data = input_data[feature_columns]

    # === Feature Scaling (Explicit Column Handling) ===
    numerical_cols = [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'Department', 'salary',
        'is_very_low_project_count', 'is_high_project_count',
        'very_high_evaluation', 'low_salary_high_hours',
        'low_satisfaction_no_promo', 'workload_index', 'within_first_3_years'
    ]
    input_scaled = scaler.transform(input_data[numerical_cols])

    # === Prediction ===
    prediction = model.predict(input_scaled)[0]

    # === Prediction Probability ===
    try:
        class_index = list(model.classes_).index(1)
        prediction_proba = model.predict_proba(input_scaled)[0][class_index]
    except (AttributeError, IndexError, ValueError):
        prediction_proba = 0.0  # Fallback

    # === Display Result (styled) ===
    if prediction == 1:
        st.error(f"🚨 This employee is likely to leave. (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ This employee is likely to stay. (Probability of leaving: {prediction_proba:.2f})")

# === FOOTER WITH GITHUB LINK ===
st.markdown(
    '<footer><small>Built by <a href="https://github.com/Phenomkay" style="color: navy; text-decoration: none;" target="_blank">Caleb Osagie</a></small></footer>',
    unsafe_allow_html=True
)