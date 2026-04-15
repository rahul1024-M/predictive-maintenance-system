import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("../models/failure_model.pkl")

st.title("Predictive Maintenance System")

st.write("Enter machine sensor values to predict failure.")

air_temp = st.number_input("Air Temperature")
process_temp = st.number_input("Process Temperature")
speed = st.number_input("Rotational Speed")
torque = st.number_input("Torque")
tool_wear = st.number_input("Tool Wear")

if st.button("Predict Failure"):
    data = np.array([[air_temp, process_temp, speed, torque, tool_wear]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠ Machine Failure Likely")
    else:
        st.success("✅ Machine Operating Normally")
