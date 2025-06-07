import pandas as pd
import numpy as np
import os

# Paths (adjust as needed)
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/your_data.csv')  # Update filename
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed_data.csv')

def load_data(path=RAW_DATA_PATH):
    """Load raw customer data from CSV."""
    return pd.read_csv(path)

def feature_engineering(df):
    """Perform basic feature engineering for churn prediction."""
    # Example: Create tenure group
    df['tenure_group'] = pd.cut(df['tenure_months'], bins=[0, 6, 12, 24, 36], labels=['0-6', '7-12', '13-24', '25-36'])
    # Example: Normalize charges
    df['monthly_charges_norm'] = (df['monthly_charges'] - df['monthly_charges'].mean()) / df['monthly_charges'].std()
    df['total_charges_norm'] = (df['total_charges'] - df['total_charges'].mean()) / df['total_charges'].std()
    return df

def save_processed_data(df, path=PROCESSED_DATA_PATH):
    """Save processed data to CSV."""
    df.to_csv(path, index=False)

def main():
    df = load_data()
    df = feature_engineering(df)
    save_processed_data(df)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
