import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("Hybrid_vehicle_predection_model.pkl")

st.set_page_config(page_title="Hybrid Vehicle Purchase Prediction", layout="centered")

st.title("🚗 Hybrid Vehicle Purchase Prediction App")
st.write("Enter your details below to predict the likelihood (%) of purchasing a hybrid vehicle.")
# Age mapping (categorical)
age_options = {
    "1 → 21–30": 1,
    "2 → 31–40": 2,
    "3 → 41–50": 3,
    "4 → 51–60": 4,
    "5 → 60-70": 5
}

age_label = st.selectbox("Select Age Group", list(age_options.keys()))
age = age_options[age_label]
education = st.selectbox("Education Level", [1, 2, 3, 4, 5], help="1=Low, 5=High")
income = st.number_input("Annual Income (in Lakhs)", min_value=1.0, max_value=100.0, value=10.0)
vehicle_ownership = st.selectbox("Number of Vehicles Owned", [0, 1, 2, 3, 4])
# Predict button
if st.button("Predict"):
    # Prepare input as DataFrame
    input_data = pd.DataFrame([[age, education, income, vehicle_ownership]],
                              columns=["Age", "Education", "Income", "Vehicle Ownership"])
    
    # Make prediction
    prediction = model.predict(input_data)[0] * 100
    st.success(f"🔮 Predicted Likelihood of Purchase: {prediction:.2f}%")

# Optional: Add footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn")
