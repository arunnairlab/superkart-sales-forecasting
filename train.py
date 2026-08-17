"""
Trains the SuperKart sales-forecasting model and saves it as superkart_model.joblib.

We don't commit the trained model binary to git (it's a derived artifact, ~15MB,
and doesn't belong in version control). Instead, run this script once after
cloning the repo to (re)generate backend/superkart_model.joblib before building
the Docker images.

Usage (from the repo root):
    pip install -r backend/requirements.txt
    python train.py
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "SuperKart.csv"
MODEL_OUTPUT_PATH = "backend/superkart_model.joblib"

# ---- Load raw data ----
data = pd.read_csv(DATA_PATH)

# ---- Data cleaning / feature engineering (matches the project notebook) ----
data.Product_Sugar_Content.replace(to_replace=["reg"], value=["Regular"], inplace=True)

data["Product_Id_char"] = data["Product_Id"].str[:2]
data["Store_Age_Years"] = 2025 - data.Store_Establishment_Year

perishables = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"]


def categorize(product_type):
    return "Perishables" if product_type in perishables else "Non Perishables"


data["Product_Type_Category"] = data["Product_Type"].apply(categorize)

# Drop columns not used as model features
data = data.drop(["Product_Id", "Product_Type", "Store_Id", "Store_Establishment_Year"], axis=1)

# ---- Train / test split ----
X = data.drop("Product_Store_Sales_Total", axis=1)
y = data["Product_Store_Sales_Total"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1, shuffle=True
)

# ---- Preprocessing pipeline ----
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# NOTE: make_column_transformer defaults to remainder='drop', which would silently
# drop all numeric predictors not listed in categorical_features. We explicitly pass
# remainder='passthrough' so numeric features (Product_MRP, Product_Weight,
# Product_Allocated_Area, Store_Age_Years) flow through into the model.
preprocessor = make_column_transformer(
    (Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    remainder="passthrough",
)

# ---- Model: tuned Random Forest (best of 6 candidate models screened in the notebook) ----
# Hyperparameters below are the best ones found via GridSearchCV in the project notebook
# (scoring="r2", cv=3) over:
#   max_depth: [None, 10, 15], max_features: [0.7, 1.0],
#   n_estimators: [100, 200], min_samples_leaf: [1, 3]
rf_tuned = RandomForestRegressor(
    random_state=1,
    max_depth=None,
    max_features=1.0,
    n_estimators=100,
    min_samples_leaf=3,
)
rf_tuned = make_pipeline(preprocessor, rf_tuned)
rf_tuned.fit(X_train, y_train)

train_r2 = rf_tuned.score(X_train, y_train)
test_r2 = rf_tuned.score(X_test, y_test)
print(f"Train R^2: {train_r2:.4f}")
print(f"Test R^2:  {test_r2:.4f}")

joblib.dump(rf_tuned, MODEL_OUTPUT_PATH)
print(f"Model saved to {MODEL_OUTPUT_PATH}")
