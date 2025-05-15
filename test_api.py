import requests
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
url = "http://localhost:5000/predict"


def generate_test_data():
    t = np.linspace(0, 10, 250)

    ppg = (
        np.sin(2 * np.pi * t)
        + 12 * np.sin(9 * np.pi * t)
        + np.random.normal(0, 0.1, 250)
    )

    ecg = (
        np.sin(2 * np.pi * t)
        + 0.1 * np.sin(16 * np.pi * t)
        + np.random.normal(0, 0.1, 250)
    )
    return ppg.tolist(), ecg.tolist()


ppg_data, ecg_data = generate_test_data()
data = {"ppg": ppg_data, "ecg": ecg_data}

logger.info("Sending request to API...")
try:
    # Make the POST request with explicit headers
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=data, headers=headers, timeout=30)

    # Print the response
    logger.info(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        logger.info("Response received successfully")
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
    else:
        logger.error(f"Error Response: {response.text}")
        print(f"\nError Response: {response.text}")
except requests.exceptions.Timeout:
    logger.error("Request timed out")
    print("Error: Request timed out")
except requests.exceptions.ConnectionError as e:
    logger.error(f"Connection error: {str(e)}")
    print(
        f"Error: Could not connect to the server. Make sure the Flask server is running."
    )
except requests.exceptions.RequestException as e:
    logger.error(f"Request error: {str(e)}")
    print(f"Error making request: {str(e)}")
