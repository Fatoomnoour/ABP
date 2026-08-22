import numpy as np

from abp_service.app import create_app


class IdentityScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class FakeModel:
    def predict(self, features, verbose=0):
        assert features.shape == (1, 250, 2)
        return np.array([[120.5]], dtype=np.float32)


class FakeBundle:
    sample_size = 250

    def __init__(self):
        self.model = FakeModel()
        self.scaler_x = IdentityScaler()
        self.scaler_y = IdentityScaler()

    def predict(self, payload):
        from abp_service.preprocessing import prepare_features

        features = prepare_features(payload, self.scaler_x, self.sample_size)
        return self.model.predict(features)


def valid_payload():
    return {"ppg": [0.1] * 250, "ecg": [0.2] * 250}


def test_home_reports_service_status():
    client = create_app(model_bundle=FakeBundle()).test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_predict_returns_mean_abp():
    client = create_app(model_bundle=FakeBundle()).test_client()

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    assert response.get_json() == {
        "predicted_mean_abp": 120.5,
        "unit": "mmHg",
    }


def test_predict_rejects_invalid_payload():
    client = create_app(model_bundle=FakeBundle()).test_client()

    response = client.post("/predict", json={"ppg": [0.1] * 250})

    assert response.status_code == 400
    assert "Missing required field" in response.get_json()["error"]


def test_predict_returns_503_when_model_is_unavailable():
    client = create_app(model_bundle=None, load_model_on_startup=False).test_client()

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 503
    assert response.get_json()["error"] == "Model is not available"
