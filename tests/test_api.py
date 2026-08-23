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


class FailingBundle:
    def predict(self, payload):
        raise RuntimeError("private model failure")


def test_predict_rejects_invalid_json(client):
    response = client.post(
        "/predict",
        data="{not-json}",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "Request body must be valid JSON"}


def test_predict_rejects_non_object_json(client):
    response = client.post("/predict", json=[0.1] * 250)
    assert response.status_code == 400
    assert "JSON object" in response.get_json()["error"]


@pytest.mark.parametrize(
    "field,value,error_text",
    [
        ("ppg", "not-a-list", "must be a JSON list"),
        ("ecg", ["bad"] * 250, "only numeric values"),
        ("ppg", [float("nan")] + [0.1] * 249, "only finite values"),
    ],
)
def test_predict_rejects_invalid_signal_values(client, field, value, error_text):
    payload = {"ppg": [0.1] * 250, "ecg": [0.2] * 250}
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert error_text in response.get_json()["error"]


def test_predict_hides_unexpected_model_errors():
    client = create_app(model_bundle=FailingBundle()).test_client()
    response = client.post("/predict", json={"ppg": [0.1] * 250, "ecg": [0.2] * 250})
    assert response.status_code == 500
    assert response.get_json() == {"error": "Prediction failed"}
    assert "private model failure" not in response.get_data(as_text=True)
