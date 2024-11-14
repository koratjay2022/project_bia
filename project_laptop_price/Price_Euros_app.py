import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("../")
from Utils.common_function import common_Fucnation,train_and_evaluate_model,train_and_evaluate_multiple_models
from unique_date_extract import unique_company, unique_typeName, unique_ScreenResolution, unique_Ram
from unique_date_extract import unique_Cpu, unique_Gpu, unique_Memory, unique_OpSys

# Load the trained Random Forest model
model = load('random_forest_model.joblib')
comm_fun = common_Fucnation()

# Create a Streamlit app
st.title("Customer Churn Prediction App")

st.header("Enter Customer Information")

company = st.selectbox("Select Company", (unique_company))
typeName = st.selectbox("Select Type Name", (unique_typeName))
inches = st.number_input('Enter screen size in inches', min_value=10.0, max_value=20.0, value=15.0, step=0.1, format="%.1f")
screenResolution = st.selectbox("Select Screen Resolution", (unique_ScreenResolution))
cpu = st.selectbox("Select CPU", (unique_Cpu))
Ram = st.select_slider("Select RAM", (unique_Ram))
memory = st.select_slider("Select Memory", (unique_Memory))
gpu = st.selectbox("Select GPU", (unique_Gpu))
OpSys = st.select_slider("Select OpSys", (unique_OpSys))
weight = st.number_input('Enter weight in kg', min_value=1.0, max_value=4.0, value=2.0, step=0.1, format="%.1f")

import re

match = re.search(r'\d+x\d+', screenResolution)
if match:
    resolution = match.group()
    resolution_width, resolution_height = map(int, resolution.split('x'))
else:
    st.error("Invalid screen resolution format. Please select a valid option.")
    st.stop()
    
cpu_brand = cpu.split()[0]
cpu_type_match = re.search(r'(Core \w+\d|Atom|FX|Celeron|E-Series|Ryzen|A\d+-Series|Pentium|Xeon|Core M)', cpu)
cpu_type = cpu_type_match.group() if cpu_type_match else None
cpu_clock_speed_match = re.search(r'(\d+\.\d+GHz|\d+GHz)', cpu)
cpu_clock_speed = cpu_clock_speed_match.group() if cpu_clock_speed_match else None

# label_encode
encoded_company = comm_fun.label_encode(data=pd.DataFrame({'Company': [company]}), column="Company").iloc[0]
encoded_cpu = comm_fun.label_encode(data=pd.DataFrame({'Cpu': [cpu]}), column="Cpu").iloc[0]
encoded_typeName = comm_fun.label_encode(data=pd.DataFrame({'TypeName': [typeName]}), column="TypeName").iloc[0]
encoded_gpu = comm_fun.label_encode(data=pd.DataFrame({'GPU': [gpu]}), column="GPU").iloc[0]
encoded_opSys = comm_fun.label_encode(data=pd.DataFrame({'OpSys': [OpSys]}), column="OpSys").iloc[0]
encoded_memory = comm_fun.label_encode(data=pd.DataFrame({'Memory': [OpSys]}), column="Memory").iloc[0]
encoded_cpu_brand = comm_fun.label_encode(data=pd.DataFrame({'CPU_Brand': [cpu_brand]}), column="CPU_Brand").iloc[0]
encoded_cpu_type = comm_fun.label_encode(data=pd.DataFrame({'CPU_Type': [cpu_type]}), column="CPU_Type").iloc[0]
encoded_cpu_clock_speed = float(cpu_clock_speed[:-3]) if cpu_clock_speed else None


# list of selected features
selected_features = [
    inches, encoded_cpu,Ram, encoded_memory, encoded_gpu, encoded_opSys, weight,
    encoded_company, encoded_typeName, resolution_width, resolution_height,
    encoded_cpu_brand, encoded_cpu_type, encoded_cpu_clock_speed
]


# App Button
if st.button("Predict Price"):
    if None in selected_features or '' in selected_features:
        st.error("Please fill all fields before making a prediction.")
    else:
        predicted_price = model.predict([selected_features])[0]

        st.header("Predicted Price")
        st.success(f"The predicted price of the laptop is: €{predicted_price:.2f}") 