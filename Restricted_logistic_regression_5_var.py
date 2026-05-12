import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv("data/Year5.csv")


altman_cols = [
"Working_CapitalTotal_Assets",
"Retained_EarningsTotal_Assets",
"EBITTotal_Assets",
"Book_Value_of_EquityTotal_Liabilities",
"SalesTotal_Assets"
]

df[altman_cols] = df[altman_cols].apply(
    pd.to_numeric, errors="coerce"
)

X = df[[
"Working_CapitalTotal_Assets",
"Retained_EarningsTotal_Assets",
"EBITTotal_Assets",
"Book_Value_of_EquityTotal_Liabilities",
"SalesTotal_Assets"
]]


y = df["class"]   #1=bankrupt, 0=non-bankrupt

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_[0]
})


print(coef_df.sort_values(by="Coefficient", ascending=False))
