"""Professional Streamlit frontend for the ABP estimation model."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import streamlit as st

from src.ml.infer import ABPInferenceEngine
from src.ml.preprocessing import SAMPLE_SIZE

st.set_page_config(
    page_title="ABP Estimation | CNN-LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
      padding: 1.4rem 1.6rem;
      border-radius: 14px;
      background: linear-gradient(135deg, #0f2747 0%, #164e63 100%);
      color: white;
      margin-bottom: 1.25rem;
    }
    .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2.25rem; }
    .hero p { color: #d8edf5; margin: 0; font-size: 1.05rem; }
    [data-testid="stMetric"] {
      border: 1px solid #dbe5ee;
      border-radius: 10px;
      padding: .7rem;
      background: #fbfdff;
    }
    .section-label {
      color: #496579;
      font-size: .82rem;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the CNN-LSTM model and scalers...")
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


def sample_signals() -> tuple[str, str]:
    """Create deterministic, non-flat demo signals for first-time users."""
    time = np.linspace(0.0, 1.0, SAMPLE_SIZE, dtype=np.float32)
    ppg = 0.7 * np.sin(2 * np.pi * 2 * time) + 0.15 * np.sin(2 * np.pi * 7 * time)
    ecg = 0.4 * np.sin(2 * np.pi * 5 * time) + 0.08 * np.cos(2 * np.pi * 13 * time)
    return json.dumps(ppg.round(6).tolist()), json.dumps(ecg.round(6).tolist())


def zero_signal() -> str:
    """Return a valid zero-valued signal for the editable input fields."""
    return json.dumps([0.0] * SAMPLE_SIZE)


def prediction_csv(values: np.ndarray) -> str:
    """Serialize prediction values as a portable CSV download."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["sample", "predicted_abp_mmHg"])
    writer.writerows((index, float(value)) for index, value in enumerate(values))
    return output.getvalue()


if "ppg_text" not in st.session_state:
    st.session_state.ppg_text = zero_signal()
if "ecg_text" not in st.session_state:
    st.session_state.ecg_text = zero_signal()

with st.sidebar:
    st.header("ABP Estimation")
    st.caption("CNN-LSTM inference demo")
    st.markdown("**Workflow**")
    st.markdown(
        "1. Prepare two 250-sample signal windows.\n"
        "2. Validate the JSON input.\n"
        "3. Run the model inference.\n"
        "4. Download the prediction."
    )
    st.divider()
    st.markdown("**Input contract**")
    st.caption("PPG and ECG must each contain exactly 250 finite numeric values.")
    st.divider()
    st.caption("Educational portfolio project. Not a certified medical device.")

st.markdown(
    """
    <div class="hero">
      <div class="section-label" style="color:#9ed7e8">Machine Learning Demo</div>
      <h1>Arterial Blood Pressure Estimation</h1>
      <p>
        Estimate an ABP response from synchronized PPG and ECG signal windows
        using a trained CNN-LSTM model.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

notice_col, status_col = st.columns([3, 1])
with notice_col:
    st.info(
        "Public demo: do not submit personally identifiable information or use "
        "the output for diagnosis, treatment, or clinical decisions."
    )
with status_col:
    st.metric("Samples / signal", SAMPLE_SIZE)

st.subheader("Signal input")
st.caption(
    "Paste valid JSON arrays below, or load a deterministic demo signal "
    "to explore the interface."
)

button_col, clear_col, _ = st.columns([1.2, 1.0, 4.0])
with button_col:
    if st.button("Load demo signals", use_container_width=True):
        st.session_state.ppg_text, st.session_state.ecg_text = sample_signals()
        st.session_state.pop("prediction", None)
        st.rerun()
with clear_col:
    if st.button("Clear inputs", use_container_width=True):
        st.session_state.ppg_text = ""
        st.session_state.ecg_text = ""
        st.session_state.pop("prediction", None)
        st.rerun()

ppg_col, ecg_col = st.columns(2, gap="large")
with ppg_col:
    ppg_text = st.text_area(
        "PPG signal",
        key="ppg_text",
        height=220,
        help=f"JSON array with exactly {SAMPLE_SIZE} finite numeric samples.",
    )
with ecg_col:
    ecg_text = st.text_area(
        "ECG signal",
        key="ecg_text",
        height=220,
        help=f"JSON array with exactly {SAMPLE_SIZE} finite numeric samples.",
    )

run_col, help_col = st.columns([1.5, 4.5])
with run_col:
    run_prediction = st.button("Predict ABP", type="primary", use_container_width=True)
with help_col:
    st.caption(
        "Validation happens before model loading. The first prediction may take "
        "longer while the model is cached."
    )

if run_prediction:
    try:
        ppg = parse_signal(ppg_text, "PPG")
        ecg = parse_signal(ecg_text, "ECG")
        prediction = np.asarray(get_engine().predict(ppg, ecg)).reshape(-1)
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.session_state.pop("prediction", None)
    else:
        st.session_state.prediction = prediction.tolist()

if "prediction" in st.session_state:
    prediction = np.asarray(st.session_state.prediction, dtype=np.float32)
    st.divider()
    st.subheader("Prediction results")
    st.success(
        f"Prediction completed successfully with {prediction.size} output value(s)."
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Mean ABP", f"{float(np.mean(prediction)):.2f} mmHg")
    metric_cols[1].metric("Minimum", f"{float(np.min(prediction)):.2f} mmHg")
    metric_cols[2].metric("Maximum", f"{float(np.max(prediction)):.2f} mmHg")

    if prediction.size > 1:
        st.line_chart(prediction, height=320)
    else:
        st.info(
            "The current model returns one mean ABP value for this signal window, "
            "so no waveform chart is available."
        )

    download_col, csv_col = st.columns(2)
    with download_col:
        st.download_button(
            "Download prediction JSON",
            data=json.dumps(prediction.tolist(), indent=2),
            file_name="abp_prediction.json",
            mime="application/json",
            use_container_width=True,
        )
    with csv_col:
        st.download_button(
            "Download prediction CSV",
            data=prediction_csv(prediction),
            file_name="abp_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

with st.expander("API integration example"):
    st.markdown(
        "The original Flask API remains available for Docker/Render deployments "
        "through `POST /predict`."
    )
    st.code(
        "curl -X POST https://YOUR-SERVICE/predict \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '{\"ppg\": [0.1, ... 250 values], "
        "\"ecg\": [0.2, ... 250 values]}'",
        language="bash",
    )
