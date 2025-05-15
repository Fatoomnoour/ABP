import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.metrics import MeanSquaredError
import keras


@keras.saving.register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)


sample_size = 250


model = load_model("CNN_LSTM_Model_256.h5", custom_objects={"mse": mse})
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")


ppg = np.random.rand(sample_size)
ecg = np.random.rand(sample_size)

ppg = ppg.reshape(1, sample_size, 1)
ecg = ecg.reshape(1, sample_size, 1)
X = np.concatenate([ppg, ecg], axis=-1)

X_flat = X.reshape(1, -1)
X_scaled = scaler_X.transform(X_flat).reshape(1, sample_size, 2)


pred_scaled = model.predict(X_scaled)
pred_abp = scaler_y.inverse_transform(pred_scaled)

print(f"Predicted Mean ABP: {pred_abp[0][0]:.2f} mmHg")
