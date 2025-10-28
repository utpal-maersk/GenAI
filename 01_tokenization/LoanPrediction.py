# Loan Repayment Probability Prediction using Linear Regression (Simplified)
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Load Training Data
train_data = pd.read_csv("Loan_Repayment_Training_Data.csv")

print("✅ Training data loaded successfully")
#train_data.info()

# # Step 2: Map Categorical Columns to Numeric
train_data["Employment Status"] = train_data["Employment Status"].map({
    "Salaried": 1,
    "Self-Employed": 0
})

train_data["Marital Status"] = train_data["Marital Status"].map({
    "Single": 0,
    "Married": 1
})

train_data["Education Level"] = train_data["Education Level"].map({
    "High School": 0,
    "Graduate": 1,
    "Post-Graduate": 2
})
train_data = train_data.dropna()
train_data.info()

# # Step 3: Define Features (X) and Target (Y)
X =train_data[["Age", "Annual Income (LPA)", "Borrowed Amount (LPA)", 
     "Credit Score", "Loan Tenure (months)", 
     "Employment Status", "Existing Loans", 
     "Marital Status", "Education Level"]]
Y = train_data["Repayment Probability"]

# # Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X, Y)
print("✅ Model trained successfully")

# # Step 5: Save Model
joblib.dump(model, "Loan_Repayment_Model.pkl")
print("💾 Model saved as Loan_Repayment_Model.pkl")

# # # Step 6: Load Test Data
test_data = pd.read_csv("Loan_Repayment_Test_Data_NoLabel.csv")
print("✅ Test data loaded successfully" )
print(test_data.isnull().sum())
test_data.head()
test_data.info()

# Convert Employment Status
test_data["Employment Status"] = test_data["Employment Status"].map({
    "Salaried": 0,
    "Self-Employed": 1
}).fillna(-1).astype(int)

# Convert Marital Status
test_data["Marital Status"] = test_data["Marital Status"].map({
    "Single": 0,
    "Married": 1
}).fillna(-1).astype(int)

# Convert Education Level
test_data["Education Level"] = test_data["Education Level"].map({
    "High School": 0,
    "Graduate": 1,
    "Post-Graduate": 2
}).fillna(-1).astype(int)
test_data.info()


# # # Step 8: Predict Repayment Probability
test_data["Predicted Repayment Probability (%)"] = (model.predict(test_data) * 100).round(2)
test_data["Loan Eligibility"] = test_data["Predicted Repayment Probability (%)"].apply(
    lambda x: "✅ Likely to Get Loan" if x >= 40 else "❌ Unlikely to Get Loan"
)
# # # Step 9: Save Predictions to CSV
test_data.to_csv("Predicted_Loan_Repayment.csv", index=False)
print("✅ Predictions saved to Predicted_Loan_Repayment.csv successfully!")

# # # Step 10: Preview First Few Predictions
print(test_data.head())
