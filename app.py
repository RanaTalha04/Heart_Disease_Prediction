import streamlit as st
import pandas as pd
import joblib

model = joblib.load('LR_Heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction App 🩶")
st.markdown("This app predicts the likelihood of heart disease based on user inputs.")
st.markdown("Provide the following details:")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain_type = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=80, max_value=200, value=120)
cholestrol = st.number_input("Cholesterol (in mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['Y', 'N'])
rest_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
maxhr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
excercise_angina = st.selectbox("Exercise Induced Angina", ['Y', 'N']) 
old_peak = st.number_input("Old Peak (ST depression)", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down']) 

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholestrol,
        'FastingBS': 1 if fasting_bs == 'Y' else 0, 
        'MaxHR': maxhr,
        'Oldpeak': old_peak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain_type: 1,
        'RestingECG_' + rest_ecg: 1,
        'ExerciseAngina_' + excercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The model predicts that you may have heart disease. Please consult a healthcare professional for a comprehensive evaluation.")
    else:
        st.success("The model predicts that you are unlikely to have heart disease. Maintain a healthy lifestyle!")