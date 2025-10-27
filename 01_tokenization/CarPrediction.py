import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
import joblib 
#Load data train model 
data=pd.read_csv("Linear-Regression-_Training-Data_-.csv") 
Y=data["Purchase hybrid vehicle"] 
X=data[["Age","Education","Income","Vehicle Ownership"]]
 #Train the model 
model=LinearRegression() 
model.fit(X,Y)
#Saved the train model to a file for later use.
joblib.dump(model,"Hybrid_vehicle_predection_model.pkl") 
print("Model has been trained and save Successfully")
 #Load the test data 
model_test_data=pd.read_csv("Linear-Regression-Python-_Test-data_.csv") 
model_test_data.head(25) 
 #Load the trained model 
prediction_model=joblib.load("Hybrid_vehicle_predection_model.pkl") 
 #apply this model 
model_test_data["Prediction %"]=(prediction_model.predict(model_test_data)*100).round(2) 
model_test_data.head(25) 
model_test_data.to_csv("Prediction_Vehicle.csv",index=False) 
print("Prediction successfully....")