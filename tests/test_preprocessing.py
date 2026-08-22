import numpy as np
import pytest

from abp_service.preprocessing import InputValidationError, prepare_features


class IdentityScaler:
    def transform(self, values):
        return values


def payload():
    return {"ppg": [0.1] * 250, "ecg": [0.2] * 250}


def test_prepare_features_returns_model_shape():
    features = prepare_features(payload(), IdentityScaler())

    assert features.shape == (1, 250, 2)
    assert features.dtype == np.float32
    assert features[0, 0].tolist() == pytest.approx([0.1, 0.2])


def test_prepare_features_rejects_wrong_length():
    data = payload()
    data["ppg"] = [0.1] * 249

    with pytest.raises(InputValidationError, match="exactly 250"):
        prepare_features(data, IdentityScaler())


def test_prepare_features_rejects_non_finite_values():
    data = payload()
    data["ecg"][3] = float("nan")

    with pytest.raises(InputValidationError, match="finite"):
        prepare_features(data, IdentityScaler())


def test_prepare_features_rejects_missing_signal():
    with pytest.raises(InputValidationError, match="Missing required field"):
        prepare_features({"ppg": [0.1] * 250}, IdentityScaler())
