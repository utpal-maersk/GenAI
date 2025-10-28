# Loan Repayment Prediction App using Streamlit + Linear Regression
import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# ✅ MUST BE FIRST Streamlit COMMAND
# ------------------------------------------------------
st.set_page_config(page_title="Loan Repayment Predictor", layout="centered")

# ------------------------------------------------------
# 🧩 Step 1: Train Model (Cached)
# ------------------------------------------------------
@st.cache_resource
def train_model():
    train_data = pd.read_csv("Loan_Repayment_Training_Data.csv")

    # Map categorical data
    train_data["Employment Status"] = train_data["Employment Status"].map({"Salaried": 1, "Self-Employed": 0})
    train_data["Marital Status"] = train_data["Marital Status"].map({"Single": 0, "Married": 1})
    train_data["Education Level"] = train_data["Education Level"].map({"High School": 0, "Graduate": 1, "Post-Graduate": 2})
    train_data.dropna(inplace=True)

    X = train_data[[
        "Age", "Annual Income (LPA)", "Borrowed Amount (LPA)",
        "Credit Score", "Loan Tenure (months)",
        "Employment Status", "Existing Loans",
        "Marital Status", "Education Level"
    ]]
    Y = train_data["Repayment Probability"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, Y)

    joblib.dump((model, scaler), "Loan_Repayment_Model_Scaled.pkl")
    return model, scaler

# ------------------------------------------------------
# ⚙️ Step 2: Load Model
# ------------------------------------------------------
model, scaler = train_model()

# ------------------------------------------------------
# 🎨 Step 3: Streamlit UI
# ------------------------------------------------------
st.title("💰 Loan Repayment Probability Predictor")
st.markdown("Predict whether a loan applicant is likely to get a loan approval based on financial parameters.")

st.header("🧍 Applicant Details")

# Age groups mapped to approximate numeric averages
age_options = {
    "1 → 21–30": 25,
    "2 → 31–40": 35,
    "3 → 41–50": 45,
    "4 → 51–60": 55,
    "5 → 60–70": 65
}
age_group = st.selectbox("Select Age Group", list(age_options.keys()))

income = st.number_input("Annual Income (LPA)", min_value=1.0, max_value=100.0, value=25.0, step=0.5)
borrowed = st.number_input("Borrowed Amount (LPA)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
credit = st.number_input("Credit Score", min_value=300, max_value=900, value=750, step=10)
tenure = st.number_input("Loan Tenure (months)", min_value=6, max_value=240, value=36, step=6)
employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education Level", ["High School", "Graduate", "Post-Graduate"])
existing = st.number_input("Existing Loans", min_value=0, max_value=10, value=0, step=1)

# Prepare input data
input_data = pd.DataFrame([{
    "Age": age_options[age_group],
    "Annual Income (LPA)": income,
    "Borrowed Amount (LPA)": borrowed,
    "Credit Score": credit,
    "Loan Tenure (months)": tenure,
    "Employment Status": 1 if employment == "Salaried" else 0,
    "Existing Loans": existing,
    "Marital Status": 1 if marital == "Married" else 0,
    "Education Level": {"High School": 0, "Graduate": 1, "Post-Graduate": 2}[education]
}])

# ------------------------------------------------------
# 📈 Step 4: Prediction Logic
# ------------------------------------------------------
if st.button("🔍 Predict Loan Eligibility"):
    scaled_input = scaler.transform(input_data)
    prob = float(model.predict(scaled_input)[0]) * 100
    prob = max(0, min(prob, 100))  # Clamp between 0–100

    st.subheader("📊 Prediction Result")
    if prob >= 40:
        st.success(f"✅ Likely to Get Loan ({prob:.2f}%)")
    else:
        st.error(f"❌ Unlikely to Get Loan ({prob:.2f}%)")

    # ------------------------------------------------------
    # 🎨 Step 5: Visual Progress Bar (dynamic fill)
    # ------------------------------------------------------
    st.markdown("### 📈 Visual Representation")
    bar_color = "green" if prob >= 40 else "red"
    st.markdown(f"""
    <div style="background-color:#eee;border-radius:10px;width:100%;height:30px;">
      <div style="background-color:{bar_color};
                  width:{prob}%;
                  height:30px;
                  border-radius:10px;">
      </div>
    </div>
    <p style="text-align:center;font-weight:bold;">{prob:.2f}%</p>
    """, unsafe_allow_html=True)
