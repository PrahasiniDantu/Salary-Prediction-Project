import streamlit as st
import joblib
import numpy as np

# Load the updated scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('salary_model.pkl')

# Streamlit input for user details
st.title("Salary Prediction App")

name = st.text_input("Enter Your Name:")
gender = st.selectbox("Select Gender:", options=["Male", "Female", "Other"])
years_of_experience = st.number_input("Enter Years of Experience:", min_value=0)

# Prepare the input data
gender_encoded = 0 if gender == "Male" else 1 if gender == "Female" else 2  # Adjust based on your model's training
input_data = np.array([[years_of_experience, gender_encoded]])  # Ensure this matches your model's input shape

# Scale the input data
scaled_input = scaler.transform(input_data)

# Make a prediction
if st.button("Predict Salary"):
    predicted_salary = model.predict(scaled_input)
    st.write(f"Hello {name}, your predicted salary is: ${predicted_salary[0]:,.2f}")
