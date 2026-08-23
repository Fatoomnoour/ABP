import numpy as np
import pytest

from abp_service.app import create_app


class IdentityScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class FakeBundle:
    sample_size = 250

    def predict(self, payload):
        from abp_service.preprocessing import prepare_features

        features = prepare_features(payload, IdentityScaler(), self.sample_size)
        assert features.shape == (1, 250, 2)
        return np.array([[120.5]], dtype=np.float32)


@pytest.fixture
def client():
    return create_app(model_bundle=FakeBundle()).test_client()


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_predict_valid_input(client):
    payload = {"ppg": [0.1] * 250, "ecg": [0.2] * 250}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.get_json() == {
        "predicted_mean_abp": 120.5,
        "unit": "mmHg",
    }


def test_predict_missing_body(client):
    response = client.post("/predict", content_type="application/json")
    assert response.status_code == 400


def test_predict_wrong_size(client):
    payload = {"ppg": [0.1] * 100, "ecg": [0.1] * 250}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
