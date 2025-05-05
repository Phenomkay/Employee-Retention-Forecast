import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Set background image
def set_bg(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: 'Poppins', sans-serif;
        }}
        label {{
            color: navy !important;
            font-weight: 600;
        }}
        </style>
    """, unsafe_allow_html=True)

# Set background
set_bg('working_employee_2.png')

# Load fonts
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=League+Spartan&family=Poppins&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Custom header with navy color
st.markdown("""
    <h1 style='color: navy; font-family: "League Spartan", sans-serif;'>Employee Attrition Predictor</h1>
    <h3 style='color: navy; font-family: "Poppins", sans-serif;'>Will this employee leave the company?</h3>
""", unsafe_allow_html=True)

# Input fields
departments = ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management']
salary_levels = ['low', 'medium', 'high']

with st.form("prediction_form"):
    satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5)
    last_evaluation = st.number_input("Last Evaluation", min_value=0.0, max_value=1.0, value=0.5)
    number_project = st.number_input("Number of Projects", min_value=2, max_value=7, value=4)
    average_montly_hours = st.number_input("Average Monthly Hours", min_value=96, max_value=310, value=200)
    time_spend_company = st.number_input("Years Spent in Company", min_value=2, max_value=10, value=3)
    Work_accident = st.selectbox("Work Accident", [0, 1])
    promotion_last_5years = st.selectbox("Promoted in Last 5 Years", [0, 1])
    Department = st.selectbox("Department", departments)
    salary = st.selectbox("Salary Level", salary_levels)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame({
            'satisfaction_level': [satisfaction_level],
            'last_evaluation': [last_evaluation],
            'number_project': [number_project],
            'average_montly_hours': [average_montly_hours],
            'time_spend_company': [time_spend_company],
            'Work_accident': [Work_accident],
            'promotion_last_5years': [promotion_last_5years],
            'Department': [Department],
            'salary': [salary]
        })

        # Feature engineering
        input_data['is_very_low_project_count'] = (input_data['number_project'] == 2).astype(int)
        input_data['is_high_project_count'] = (input_data['number_project'] > 5).astype(int)
        input_data['very_high_evaluation'] = (input_data['last_evaluation'] > 0.9).astype(int)
        input_data['low_salary_high_hours'] = ((input_data['salary'] == 'low') & (input_data['average_montly_hours'] > 250)).astype(int)
        input_data['low_satisfaction_no_promo'] = ((input_data['satisfaction_level'] < 0.4) & (input_data['promotion_last_5years'] == 0)).astype(int)
        input_data['within_first_3_years'] = (input_data['time_spend_company'] <= 3).astype(int)
        input_data['workload_index'] = input_data['number_project'] * input_data['average_montly_hours']

        # Encode
        input_data = pd.get_dummies(input_data, columns=['Department', 'salary'], drop_first=True)

        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[expected_cols]

        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1]

        # Styled prediction output
        st.markdown(f"""
            <div style="color: navy; font-size: 20px; font-weight: bold; padding: 10px 0;">
                Prediction: {'Will Leave' if prediction == 1 else 'Will Stay'}
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="color: navy; font-size: 18px; padding-bottom: 20px;">
                Probability of leaving: {prediction_proba:.2f}
            </div>
        """, unsafe_allow_html=True)

# Footer in navy
st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 14px; color: navy;">
        Designed and Developed by 
        <a href="https://github.com/Phenomkay" target="_blank" style="text-decoration: none; color: navy;"><strong>Caleb Osagie</strong></a>
    </div>
""", unsafe_allow_html=True)
