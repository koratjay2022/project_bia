import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model
model = load('LinearRegression_model.joblib')

# Streamlit App Title
st.title("Gym Member Exercise Tracking - Calories Burned Predictor")

# Collect User Inputs for Prediction
st.header("Enter Details for Prediction")
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ("Male", "Female"))
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
max_bpm = st.number_input("Max BPM", min_value=100, max_value=220, value=180)
avg_bpm = st.number_input("Avg BPM", min_value=50, max_value=200, value=130)
resting_bpm = st.number_input("Resting BPM", min_value=40, max_value=100, value=60)
session_duration = st.number_input("Session Duration (hours)", min_value=0.1, max_value=5.0, value=1.0)
workout_type = st.selectbox("Workout Type", ("Yoga", "HIIT", "Cardio", "Strength"))
fat_percentage = st.number_input("Fat Percentage", min_value=5.0, max_value=50.0, value=20.0)
water_intake = st.number_input("Water Intake (liters)", min_value=0.5, max_value=5.0, value=2.0)
workout_frequency = st.number_input("Workout Frequency (days/week)", min_value=1, max_value=7, value=3)
experience_level = st.selectbox("Experience Level", ("1", "2", "3"))
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

# Encode inputs for model
gender = 1 if gender == "Male" else 0
workout_type_map = {"Yoga": 0, "HIIT": 1, "Cardio": 2, "Strength": 3}
workout_type = workout_type_map[workout_type]
experience_level = int(experience_level)

# Prepare input data
input_data = np.array([[age, gender, weight, height, max_bpm, avg_bpm, resting_bpm, session_duration,
                        workout_type, fat_percentage, water_intake, workout_frequency, experience_level, bmi]])


# Streamlit App Button 
if st.button("Predict Calories Burned"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")


# if st.button("Predict Price"):
#     if None in selected_features or '' in selected_features:
#         st.error("Please fill all fields before making a prediction.")
#     else:
#         predicted_price = model.predict([selected_features])[0]

#         st.header("Predicted Price")
#         st.success(f"The predicted price of the laptop is: €{predicted_price:.2f}") 
# if None in selected_features or '' in selected_features:
#     st.error("Please fill all fields before making a prediction.")
# else:
#     predicted_price = model.predict([selected_features])[0]
#     st.header("Predicted Price")
#     st.success(f"The predicted price of the laptop is: {predicted_price:.2f}") 

# Ensure all features are filled before prediction
# if None in selected_features or '' in selected_features:
#     st.error("Please fill all fields before making a prediction.")
# else:
#     # Make a prediction using the model
#     prediction = model.predict([selected_features])

#     # Display the prediction result on the main screen
#     st.header("Prediction Result")
#     if prediction[0] == 0:
#         st.success("This customer is likely to stay.")
#     else:
#         st.error("This customer is likely to churn.")
        
# how to run
# streamlit run file name
# E.g. :- streamlit run Price_euros_app.py