import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    with patch("src.api.app.load_artifacts") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[85.0, 87.0, 90.0]])
        mock_scaler_X = MagicMock()
        mock_scaler_X.transform.return_value = np.random.randn(1, 500)
        mock_scaler_y = MagicMock()
        mock_scaler_y.inverse_transform.return_value = np.array([[85.0, 87.0]])
        mock_load.return_value = (mock_model, mock_scaler_X, mock_scaler_y)
        from src.api.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "API is running"


def test_predict_valid_input(client):
    payload = {"ppg": np.random.randn(250).tolist(), "ecg": np.random.randn(250).tolist()}
    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200


def test_predict_missing_body(client):
    response = client.post("/predict", content_type="application/json")
    assert response.status_code == 400


def test_predict_wrong_size(client):
    payload = {"ppg": [0.1]*100, "ecg": [0.1]*250}
    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 400
