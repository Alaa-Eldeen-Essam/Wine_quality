# Wine Quality Prediction API

FastAPI backend for wine quality prediction using a trained neural network model.
 deployemont in railway https://web-production-2ea7d.up.railway.app/
## 📋 Features

- ✅ Single and batch prediction endpoints
- ✅ Health check and model info endpoints
- ✅ Input validation using Pydantic
- ✅ Optional API key authentication
- ✅ Request logging with rotation
- ✅ SHAP-based prediction explanations
- ✅ Auto-generated Swagger UI documentation
- ✅ CORS support for frontend integration

## 🚀 Quick Start

### 3. Batch Prediction
```bash
POST /predict_batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "data": [
    [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0],
    [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8, 0.0]
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.87,
      "class_name": "Class_1"
    },
    {
      "prediction": 2,
      "probability": 0.92,
      "class_name": "Class_2"
    }
  ]
}
```

### 4. Model Information
```bash
GET /model_info
```

**Response:**
```json
{
  "input_shape": "(None, 12)",
  "output_shape": "(None, 3)",
  "total_params": 1234,
  "trainable_params": 1234,
  "layers": [
    {
      "name": "dense",
      "type": "Dense",
      "output_shape": "(None, 64)",
      "params": 832
    }
  ],
  "expected_features": 12
}
```

### 5. SHAP Explanation
```bash
POST /explain
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]
}
```

**Response:**
```json
{
  "shap_values": [[0.12, -0.05, 0.03, ...]],
  "base_value": 0.5,
  "features": [7.4, 0.7, 0.0, ...]
}
```

## 🔐 Authentication

To enable API key authentication:

```bash
# Set environment variable
export API_KEY="your-secret-key-here"

# Start server
uvicorn main:app --reload
```

**Making authenticated requests:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-key-here" \
  -d '{"features": [...]}'
```

## 📊 Feature Names

The model expects 12 features in this order:

1. `fixed_acidity`
2. `volatile_acidity`
3. `citric_acid`
4. `residual_sugar`
5. `chlorides`
6. `free_sulfur_dioxide`
7. `total_sulfur_dioxide`
8. `density`
9. `pH`
10. `sulphates`
11. `alcohol`
12. `type_encoded` (0 for red, 1 for white)

## 🧪 Testing

### Using Python

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
data = {
    "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]
}
response = requests.post(url, json=data)
print(response.json())

# Batch prediction
url = "http://localhost:8000/predict_batch"
data = {
    "data": [
        [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0],
        [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8, 0.0]
    ]
}
response = requests.post(url, json=data)
print(response.json())
```

### Run Test Suite

```bash
python test_api.py
```

## 📝 Logging

Logs are stored in `prediction_requests.log` with automatic rotation:
- Max file size: 5 MB
- Backup count: 3 files

**Log format:**
```
2025-10-20 10:30:45 - INFO - /predict - OK - input_len=12 pred=1 prob=0.8700
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API authentication key | None (disabled) |
| `MODEL_PATH` | Path to model file | `model_4.h5` |
| `SCALER_PATH` | Path to scaler file | `scaler.pkl` |

### File Structure

```
src/
├── main.py                    # FastAPI application
├── models.py                  # Pydantic models
├── preprocessing.py           # Data preprocessing
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── model_4.h5                # Trained model
├── scaler.pkl                # Fitted scaler (optional)
└── prediction_requests.log   # Log file
```

## 🐛 Troubleshooting

### Model Not Loading

```bash
# Check if model file exists
ls -lh model_4.h5

# Check logs
tail -f prediction_requests.log

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Invalid Feature Count

Ensure your input matches the expected feature count (12 features):
```python
# Check model info
response = requests.get("http://localhost:8000/model_info")
print(response.json()["expected_features"])
```

### SHAP Not Available

```bash
# Install SHAP
pip install shap

# Restart server
uvicorn main:app --reload
```

## 🚀 Performance Tips

1. **Use batch predictions** for multiple samples (more efficient)
2. **Enable workers** in production: `--workers 4`
3. **Set up caching** for repeated predictions
4. **Use gunicorn** for production deployment
5. **Initialize SHAP explainer** at startup (done automatically)

## 📦 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Using Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${API_KEY}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please submit issues and pull requests.

## 📧 Support

For issues and questions, please open an issue on GitHub or contact the maintainers.Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Trained model file: `model_4.h5`
- Optional: scaler file `scaler.pkl`

### Installation

```bash
# Clone the repository
cd backend/

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Docker

```bash
# Build image
docker build -t wine-quality-api .

# Run container
docker run -p 8000:8000 wine-quality-api
```

## 📚 API Documentation

Once the server is running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "expected_features": 12
}
```

### 2. Single Prediction
```bash
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.87,
  "class_name": "Class_1"
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]}'
```

###