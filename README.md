# ABP Estimation API

> A portfolio ML service that demonstrates model-backed API inference for mean arterial blood pressure.

![Status](https://img.shields.io/badge/status-educational-demo---not-a-medical-device-blue)

## What it does

**PPG + ECG → validation → preprocessing → CNN-LSTM → ABP estimate**

## Tech stack

`Python · Flask · TensorFlow/Keras · Docker · Pytest`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
python app.py
```

## Project layout

The repository keeps the implementation, configuration, and supporting assets close to the workflow so the project is easy to inspect and reproduce. See the source folders and files for the detailed implementation.

## Important notes

**Status:** Educational demo — not a medical device. Use sample or synthetic data only unless the project documentation explicitly states otherwise. Review the limitations and security notes before any deployment or real-world use.

## License

See the repository license file when present. Contributions and improvements should keep the existing attribution and project history clear.
