# Test for FastAPI churn prediction API
from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_churn():
    # Example input, adjust keys to match your features
    features = {
        "tenure_months": 12,
        "monthly_charges": 50.0,
        "total_charges": 600.0,
        # Remove 'tenure_group', 'monthly_charges_norm', 'total_charges_norm' from input
        # as these are engineered and one-hot encoded in the API
    }
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "churn_probability" in response.json()
