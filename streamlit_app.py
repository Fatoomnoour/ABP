"""Streamlit demo for the ABP estimation model.

The Flask API remains available through ``app.py``. This file is the entry point
for Streamlit Community Cloud, where a public interactive demo can be deployed
without requiring a Docker service.
"""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

from src.ml.infer import ABPInferenceEngine
from src.ml.preprocessing import SAMPLE_SIZE

st.set_page_config(page_title="ABP Estimation", page_icon="📈", layout="centered")


@st.cache_resource(show_spinner="Loading the CNN-LSTM model...")
def get_engine() -> ABPInferenceEngine:
    """Load the model once per Streamlit process."""
    return ABPInferenceEngine()


def parse_signal(raw: str, name: str) -> list[float]:
    """Parse and validate a JSON array containing one biosignal."""
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a valid JSON array.") from exc

    if not isinstance(values, list):
        raise ValueError(f"{name} must be a JSON array.")
    if len(values) != SAMPLE_SIZE:
        raise ValueError(f"{name} must contain exactly {SAMPLE_SIZE} samples.")

    try:
        signal = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values only.") from exc

    if not np.isfinite(signal).all():
        raise ValueError(f"{name} must contain finite numeric values only.")
    return signal.tolist()


def default_signal() -> str:
    """Return a deterministic signal-shaped JSON example."""
    return json.dumps([0.0] * SAMPLE_SIZE)


st.title("ABP Estimation")
st.write(
    "Upload or paste 250-sample PPG and ECG signals as JSON arrays to estimate "
    "the arterial blood-pressure waveform."
)
st.info(
    "This is a public demo. Do not submit personally identifiable information "
    "or clinical decisions based on the output."
)

ppg_text = st.text_area(
    "PPG signal (JSON array, 250 values)",
    value=default_signal(),
    height=120,
)
ecg_text = st.text_area(
    "ECG signal (JSON array, 250 values)",
    value=default_signal(),
    height=120,
)

if st.button("Predict ABP", type="primary"):
    try:
        ppg = parse_signal(ppg_text, "PPG")
        ecg = parse_signal(ecg_text, "ECG")
        prediction = np.asarray(get_engine().predict(ppg, ecg)).reshape(-1)
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
    else:
        st.success(f"Prediction completed: {prediction.size} output values.")
        st.line_chart(prediction)
        st.download_button(
            "Download prediction JSON",
            data=json.dumps(prediction.tolist()),
            file_name="abp_prediction.json",
            mime="application/json",
        )
