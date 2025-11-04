import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("Bank_marketing.pkl")

st.title("🏦 Bank Marketing Campaign Prediction App")
st.write("Predict whether a customer will subscribe to a term deposit.")

# --- Input fields ---
age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job Type", [
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
    'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'
])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.selectbox("Education Level", ['primary', 'secondary', 'tertiary', 'unknown'])
balance = st.number_input("Account Balance (€)", value=1000)
housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
loan = st.selectbox("Has Personal Loan?", ['yes', 'no'])
contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
month = st.selectbox("Last Contact Month", [
    'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'
])
duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=120)
campaign = st.number_input("Number of Contacts in Campaign", min_value=1, value=1)

# --- Create input DataFrame ---
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'duration': [duration],
    'campaign': [campaign]
})

# Match model columns (same encoding used during training)
input_encoded = pd.get_dummies(input_data)
# Re-align with model's columns (fill missing with 0)
model_features = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# --- Prediction ---
if st.button("Predict"):
    prob = model.predict_proba(input_encoded)[0][1] * 100
    pred = model.predict(input_encoded)[0]

    if pred == 1:
        st.success(f"✅ Customer is likely to subscribe! (Probability: {prob:.2f}%)")
    else:
        st.error(f"❌ Customer may not subscribe. (Probability: {prob:.2f}%)")
