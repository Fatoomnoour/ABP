"""Send a deterministic sample request to a running ABP API."""

from __future__ import annotations

import argparse
import json

import numpy as np
import requests


def sample_payload(sample_size: int = 250) -> dict[str, list[float]]:
    rng = np.random.default_rng(42)
    time = np.linspace(0, 10, sample_size)
    ppg = np.sin(2 * np.pi * time) + 0.1 * rng.normal(size=sample_size)
    ecg = np.sin(2 * np.pi * time) + 0.1 * rng.normal(size=sample_size)
    return {"ppg": ppg.tolist(), "ecg": ecg.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:5000/predict")
    args = parser.parse_args()

    response = requests.post(args.url, json=sample_payload(), timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
