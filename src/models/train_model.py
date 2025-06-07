import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# Paths (adjust as needed)
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed_data.csv')
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../../data/churn_model.pkl')


def load_processed_data(path=PROCESSED_DATA_PATH):
    """Load processed data for modeling."""
    return pd.read_csv(path)


def train_model(df):
    """Train a Random Forest model for churn prediction."""
    # Features and target
    X = df.drop(['customer_id', 'churn'], axis=1, errors='ignore')
    y = df['churn']
    # Ensure tenure_group has all categories
    from pandas.api.types import CategoricalDtype
    tenure_group_type = CategoricalDtype(categories=['0-6', '7-12', '13-24', '25-36'])
    if 'tenure_group' in X.columns:
        X['tenure_group'] = X['tenure_group'].astype(tenure_group_type)
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['tenure_group'])
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # Evaluation
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return clf


def save_model(model, path=MODEL_OUTPUT_PATH):
    import joblib
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    df = load_processed_data()
    model = train_model(df)
    save_model(model)


if __name__ == "__main__":
    main()
