"""
API endpoint tests for ABP Estimation Flask app.
Run with: pytest tests/test_api.py -v
"""

import json
import numpy as np
import pytest

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a Flask test client."""
    from src.api.app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _make_payload(ppg_size=250, ecg_size=250):
    """Generate a valid test payload."""
    return {
        "ppg": np.random.randn(ppg_size).tolist(),
        "ecg": np.random.randn(ecg_size).tolist(),
    }


# ─── Health check ─────────────────────────────────────────────────────────────

def test_health_check(client):
    """GET / should return 200 with status info."""
    response = client.get("/")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "API is running"
    assert "/predict" in data["endpoints"]


# ─── Valid prediction ──────────────────────────────────────────────────────────

def test_predict_valid_input(client):
    """POST /predict with valid input should return 200 and predicted_abp."""
    payload = _make_payload()
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "predicted_abp" in data
    assert isinstance(data["predicted_abp"], list)


# ─── Input validation errors ──────────────────────────────────────────────────

def test_predict_missing_body(client):
    """POST /predict with no body should return 400."""
    response = client.post("/predict", content_type="application/json")
    assert response.status_code == 400


def test_predict_missing_ecg(client):
    """POST /predict missing 'ecg' field should return 400."""
    payload = {"ppg": np.random.randn(250).tolist()}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_predict_wrong_ppg_size(client):
    """POST /predict with wrong PPG size should return 400."""
    payload = _make_payload(ppg_size=100, ecg_size=250)
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "250" in data["error"]


def test_predict_wrong_ecg_size(client):
    """POST /predict with wrong ECG size should return 400."""
    payload = _make_payload(ppg_size=250, ecg_size=50)
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_predict_empty_lists(client):
    """POST /predict with empty arrays should return 400."""
    payload = {"ppg": [], "ecg": []}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
