import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the trained Random Forest model
model = load('Random_Forest_model.joblib')

# Streamlit UI
st.title('Customer Feedback Satisfaction Prediction')

st.write("""
    This app predicts customer satisfaction based on various features such as gender, country, product quality, and feedback score.
""")

# Show Feature Importance
st.header("Enter Details for Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
country = st.selectbox("Country", ["UK", "USA", "France", "Germany", "Canada"])
income = st.number_input("Income", min_value=1000, max_value=200000, value=50000)
product_quality = st.slider("Product Quality", 1, 5)
service_quality = st.slider("Service Quality", 1, 10)
purchase_frequency = st.slider("Purchase Frequency", 1, 12, value=5)
feedback_score = st.selectbox("Feedback Score", ["Low", "Medium", "High"])
loyalty_level = st.selectbox("Loyalty Level", ["Bronze", "Gold", "Silver"])

# Additional user inputs for missing features

# Select feature
feature_columns = ['Gender', 'Country', 'ProductQuality', 'ServiceQuality', 'FeedbackScore', 'LoyaltyLevel', 'Age', 'Income', 'PurchaseFrequency']

# Create a DataFrame for user input
user_input = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 2],
    'Country': [{"UK": 1, "USA": 2, "France": 3, "Germany": 4, "Canada": 5}[country]],
    'ProductQuality': [product_quality],
    'ServiceQuality': [service_quality],
    'FeedbackScore': [{"Low": 1, "Medium": 2, "High": 3}[feedback_score]],
    'LoyaltyLevel': [{"Bronze": 1, "Gold": 2, "Silver": 3}[loyalty_level]],
    'Age': [age],
    'Income': [income],
    'PurchaseFrequency': [purchase_frequency]
}, columns=feature_columns)

# Ensure the order of features in user input matches the model's training order
user_input = user_input[feature_columns]

# Prediction triggered by button
if st.button('Predict'):
    # Make the prediction
    prediction = model.predict(user_input)
    st.write(f"**Predicted Satisfaction Score:** {prediction[0]:.2f}")
else:
    st.write("Click 'Predict' to get the Satisfaction Score prediction.")
