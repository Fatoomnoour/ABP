"""Run a smoke test against a running ABP Estimation API."""

from __future__ import annotations

import argparse

import requests
from infer_request import sample_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:5000")
    args = parser.parse_args()

    health = requests.get(f"{args.base_url}/", timeout=10)
    health.raise_for_status()
    prediction = requests.post(
        f"{args.base_url}/predict", json=sample_payload(), timeout=30
    )
    prediction.raise_for_status()
    print("Health:", health.json())
    print("Prediction:", prediction.json())


if __name__ == "__main__":
    main()
