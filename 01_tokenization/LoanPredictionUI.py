import streamlit as st
import numpy as np
import joblib

# -------------------------------------------------
# 1️⃣ Page Configuration — MUST BE FIRST STREAMLIT COMMAND
# -------------------------------------------------
st.set_page_config(page_title="💰 Loan Eligibility Predictor", layout="centered")

# -------------------------------------------------
# 2️⃣ Load the trained model and scaler
# -------------------------------------------------
model, scaler = joblib.load("Loan_Repayment_Model_Scaled.pkl")

# -------------------------------------------------
# 3️⃣ Helper Dictionaries for Input Encoding
# -------------------------------------------------
age_options = {
    "1 → 21–30": 1,
    "2 → 31–40": 2,
    "3 → 41–50": 3,
    "4 → 51–60": 4,
    "5 → 60–70": 5
}

employment_options = {"Salaried": 1, "Self-Employed": 2, "Unemployed": 3}
marital_options = {"Single": 1, "Married": 2, "Divorced": 3, "Widowed": 4}
education_options = {"Graduate": 1, "Post-Graduate": 2, "Doctorate": 3, "Others": 4}

# -------------------------------------------------
# 4️⃣ Streamlit UI
# -------------------------------------------------
st.title("💼 Loan Eligibility Prediction App")

st.markdown("### 🧾 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Select Age Group", list(age_options.keys()))
    annual_income = st.number_input("Annual Income (LPA)", min_value=0.0, max_value=100.0, value=10.0)
    borrowed_amount = st.number_input("Borrowed Amount (LPA)", min_value=0.0, max_value=100.0, value=5.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750)
    loan_tenure = st.number_input("Loan Tenure (months)", min_value=6, max_value=240, value=36)

with col2:
    employment_status = st.selectbox("Employment Status", list(employment_options.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_options.keys()))
    education_level = st.selectbox("Education Level", list(education_options.keys()))
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)

# -------------------------------------------------
# 5️⃣ Prepare input features
# -------------------------------------------------
input_data = np.array([[
    age_options[age],
    annual_income,
    borrowed_amount,
    credit_score,
    loan_tenure,
    employment_options[employment_status],
    existing_loans,
    marital_options[marital_status],
    education_options[education_level]
]])

# Scale the data
input_scaled = scaler.transform(input_data)

# -------------------------------------------------
# 6️⃣ Prediction
# -------------------------------------------------
if st.button("🔍 Predict Loan Eligibility"):
    prediction = model.predict(input_scaled)[0]
    prediction_percent = prediction * 100

    st.markdown("### 📊 Prediction Result")
    st.metric(label="Predicted Repayment Probability", value=f"{prediction_percent:.2f}%")

    # Visual feedback
    if prediction_percent > 40:
        st.success("✅ Eligible for Loan — High repayment likelihood!")
        st.progress(int(prediction_percent))
    elif 20 <= prediction_percent <= 40:
        st.warning("⚠️ Moderate chance — borderline eligibility.")
        st.progress(int(prediction_percent))
    else:
        st.error("❌ Unlikely to Get Loan — Low repayment probability.")
        st.progress(int(prediction_percent))

# -------------------------------------------------
# 7️⃣ Footer / Notes
# -------------------------------------------------
st.markdown("---")
st.caption("💡 This app uses a trained Linear Regression model to estimate loan repayment probability based on applicant details.")
