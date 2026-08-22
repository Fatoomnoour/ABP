"""Train the CNN-LSTM ABP model from the source MAT dataset.

The checked-in notebook remains available for exploration, while this script
provides a repeatable training entry point with train-only scaler fitting.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_windows(
    data_file: Path, sample_size: int = 250, max_records: int | None = 1000
):
    """Load PPG, ECG, and mean-ABP windows from the MAT file used in training."""

    mat = loadmat(data_file)["p"]
    record_count = min(mat.shape[1], max_records) if max_records else mat.shape[1]
    ppg, ecg, abp = [], [], []

    for index in range(record_count):
        record = mat[0, index]
        for offset in range(record.shape[1] // sample_size):
            start = offset * sample_size
            end = start + sample_size
            ppg.append(record[0, start:end])
            abp.append(record[1, start:end])
            ecg.append(record[2, start:end])

    ppg_array = np.asarray(ppg, dtype=np.float32)
    ecg_array = np.asarray(ecg, dtype=np.float32)
    abp_array = np.asarray(abp, dtype=np.float32)
    features = np.stack((ppg_array, ecg_array), axis=-1)
    targets = abp_array.mean(axis=1).reshape(-1, 1)
    return features, targets


def build_model(sample_size: int = 250):
    """Build the CNN-LSTM architecture used by the original notebook."""

    from tensorflow.keras import Sequential, layers
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            layers.Input(shape=(sample_size, 2)),
            layers.Conv1D(256, 5, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            layers.Conv1D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            layers.LSTM(256),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=5e-4), loss="mse", metrics=["mae"])
    return model


def train(data_file: Path, output_dir: Path, epochs: int, batch_size: int) -> dict:
    """Train, evaluate, and export the model and fitted preprocessing artifacts."""

    sample_size = 250
    features, targets = load_windows(data_file, sample_size=sample_size)
    indices = np.arange(features.shape[0])
    train_indices, test_indices = train_test_split(
        indices, test_size=0.3, random_state=42
    )

    scaler_x = StandardScaler().fit(
        features[train_indices].reshape(len(train_indices), -1)
    )
    scaler_y = StandardScaler().fit(targets[train_indices])
    scaled_features = scaler_x.transform(
        features.reshape(features.shape[0], -1)
    ).reshape(
        features.shape
    )
    scaled_targets = scaler_y.transform(targets)

    model = build_model(sample_size)
    model.fit(
        scaled_features[train_indices],
        scaled_targets[train_indices],
        validation_data=(scaled_features[test_indices], scaled_targets[test_indices]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    predictions = scaler_y.inverse_transform(
        model.predict(scaled_features[test_indices], verbose=0)
    )
    actual = targets[test_indices]
    metrics = {
        "rmse_mmHg": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
        "test_samples": int(len(test_indices)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "CNN_LSTM_Model_256.h5")
    for name, scaler in (("scaler_X.pkl", scaler_x), ("scaler_y.pkl", scaler_y)):
        with (output_dir / name).open("wb") as file:
            pickle.dump(scaler, file)

    logger.info("Saved model artifacts to %s", output_dir)
    logger.info("Evaluation: %s", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    train(args.data_file, args.output_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
