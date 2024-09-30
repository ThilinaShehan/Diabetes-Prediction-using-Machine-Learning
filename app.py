import streamlit as st
import numpy as np
import joblib

# Load your trained machine learning model from a .sav file
classifier = joblib.load('model.sav')

st.title("Patient Diabetes Prediction")

st.write("Enter the following information:")

# Define a CSS class for styling text input fields
st.markdown(
    """
    <style>
        .text-input {
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .success-message {
            padding: 10px;
            color: #4CAF50;
            border-radius: 5px;
            text-align: center;
            font-size:20px;
            font-weight:bold;
        }
        .danger-message {
            padding: 10px;
            color: #FF5733;
            border-radius: 5px;
            text-align: center;
            font-size:20px;
            font-weight:bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=100)
glucose = st.number_input("Glucose", min_value=0, max_value=500)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=1000)
bmi = st.text_input("BMI", value="0.000", help='Enter BMI', key="bmi_input")
bmi = float(bmi) if bmi else 0.0  # Convert to float, handle empty input
bmi = round(bmi, 3)  # Round to 3 decimal places

# Use a text input for Diabetes Pedigree Function
diabetes_pedigree = st.text_input("Diabetes Pedigree Function", value="0.000", help='Enter Diabetes Pedigree Function', key="diabetes_input")
diabetes_pedigree = float(diabetes_pedigree) if diabetes_pedigree else 0.0  # Convert to float, handle empty input
diabetes_pedigree = round(diabetes_pedigree, 3)  # Round to 3 decimal places
age = st.number_input("Age", min_value=0, max_value=120)

input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)

if st.button("Predict"):
    prediction = classifier.predict(input_data)

    if prediction == 0:
        st.markdown('<div class="success-message">The Patient is not Diabetic.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="danger-message">The Patient is Diabetic.</div>', unsafe_allow_html=True)
