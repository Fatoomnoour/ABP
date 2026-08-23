# ABP Estimation API — Reference

## Base URL
```
http://localhost:5000
```

---

## Endpoints

### `GET /`
**Health check** — confirms API is running and lists available endpoints.

**Response `200`:**
```json
{
  "status": "API is running",
  "version": "1.0.0",
  "endpoints": {
    "/predict": {
      "method": "POST",
      "description": "Predict ABP waveform from PPG and ECG signals",
      "input": {
        "ppg": "List of 250 float values",
        "ecg": "List of 250 float values"
      },
      "output": {
        "predicted_abp": "Nested list of predicted ABP values (mmHg)"
      }
    }
  }
}
```

---

### `POST /predict`
**Predict ABP** from PPG and ECG biosignals.

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ppg` | `float[]` | ✅ | 250 PPG signal samples |
| `ecg` | `float[]` | ✅ | 250 ECG signal samples |

**Success Response `200`:**
```json
{
  "predicted_abp": [[85.3, 87.1, 90.4, 92.0, ...]]
}
```

**Error Responses:**

| Code | Condition | Response |
|------|-----------|----------|
| `400` | Empty body | `{"error": "Request body is empty or not valid JSON"}` |
| `400` | Missing field | `{"error": "Missing required fields: ['ecg']"}` |
| `400` | Wrong size | `{"error": "Both 'ppg' and 'ecg' must contain exactly 250 values."}` |
| `500` | Server error | `{"error": "Internal server error. Please check logs."}` |

---

## Signal Requirements

| Parameter | Value |
|-----------|-------|
| Sampling rate | 125 Hz (recommended) |
| Window size | 250 samples (= 2 seconds at 125 Hz) |
| PPG range | Normalized float values |
| ECG range | Normalized float values |

---

## Example — Python Client

```python
import requests
import numpy as np

BASE_URL = "http://localhost:5000"

def predict_abp(ppg_signal: list, ecg_signal: list) -> dict:
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"ppg": ppg_signal, "ecg": ecg_signal},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

# Usage
ppg = np.random.randn(250).tolist()
ecg = np.random.randn(250).tolist()
result = predict_abp(ppg, ecg)
print(result["predicted_abp"])
```

## Example — cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"ppg\": $(python -c \"import numpy as np; print(np.random.randn(250).tolist())\"), \
       \"ecg\": $(python -c \"import numpy as np; print(np.random.randn(250).tolist())\")}"
```
