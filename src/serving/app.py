from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../data/churn_model.pkl')

# Load model at startup
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running."}

@app.post("/predict")
def predict_churn(features: dict):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    # Convert input features to DataFrame
    X = pd.DataFrame([features])
    # --- Feature engineering (same as in feature_engineering.py) ---
    # Tenure group
    tenure_group_type = CategoricalDtype(categories=['0-6', '7-12', '13-24', '25-36'])
    X['tenure_group'] = pd.cut(X['tenure_months'], bins=[0, 6, 12, 24, 36], labels=tenure_group_type.categories).astype(tenure_group_type)
    # Normalize charges using training data mean/std
    sample_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/processed_data.csv'))
    X['monthly_charges_norm'] = (X['monthly_charges'] - sample_df['monthly_charges'].mean()) / sample_df['monthly_charges'].std()
    X['total_charges_norm'] = (X['total_charges'] - sample_df['total_charges'].mean()) / sample_df['total_charges'].std()
    # One-hot encoding for all categorical columns (including tenure_group)
    X = pd.get_dummies(X, columns=['tenure_group'])
    # Align columns with model input
    X_train = pd.get_dummies(sample_df.drop(['customer_id', 'churn'], axis=1, errors='ignore'))
    for col in X_train.columns:
        if col not in X.columns:
            X[col] = 0
    X = X[X_train.columns]
    X = X.reindex(columns=X_train.columns, fill_value=0)
    # Ensure column order matches model's feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        X = X[model.feature_names_in_]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return {"churn_prediction": int(pred), "churn_probability": float(prob)}
