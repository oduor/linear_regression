import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import requests, json

df = pd.read_csv("./SalaryData.csv")

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
df_copy = train_set.copy()
test_set_full = test_set.copy()
test_set = test_set.drop(["Salary"], axis=1)
train_labels = df_copy["Salary"]
train_set_full = train_set.copy()
train_set = train_set.drop(["Salary"], axis=1)
lin_reg = LinearRegression()
lin_reg.fit(train_set, train_labels)
salary_pred = lin_reg.predict(test_set)
salary_pred

import pickle

with open("python_lin_reg_model.pkl", "wb") as file_handler:
    pickle.dump(lin_reg, file_handler)

with open("python_lin_reg_model.pkl", "rb") as file_handler:
    loaded_pickle = pickle.load(file_handler)

loaded_pickle



BASE_URL = "http://localhost:12344"

joblib.dump(lin_reg, "linear_regression_model.pkl")

joblib.dump(train_set, "training_data.pkl")
joblib.dump(train_labels, "training_labels.pkl")


years_exp = {"yearsOfExperience": 8}
response = requests.post("{}/predict".format(BASE_URL), json = years_exp)

response.json()
