from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import SinglePredictionRequest, BatchPredictionRequest, PredictionResponse, BatchPredictionResponse
from preprocessing import preprocess_single, preprocess_batch, load_scaler_if_exists
import numpy as np
import os
import traceback
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import joblib

app = FastAPI(title="Model 4 Prediction API",
              description="FastAPI app serving model_4.h5 with prediction endpoints",
              version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API key auth (optional) - set environment variable API_KEY to enable
API_KEY = os.environ.get("API_KEY", None)

def get_api_key(request: Request):
    if API_KEY is None:
        return True
    key = request.headers.get("x-api-key")
    if not key or key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key")
    return True

# Logging
log_file = os.path.join(os.getcwd(), "prediction_requests.log")
handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger("prediction_api")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

MODEL_PATH = "model_4.h5"
SCALER_PATH = "scaler.pkl"

@app.on_event("startup")
def load_model_on_startup():
    """Load the Keras model and optional scaler on app startup."""
    try:
        # Lazy import to avoid heavy imports at module import time
        from tensorflow.keras.models import load_model
        app.state.model = load_model(MODEL_PATH)
        app.state.model_loaded = True
        logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        app.state.model = None
        app.state.model_loaded = False
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())

    # Load scaler if exists
    try:
        if os.path.exists(SCALER_PATH):
            app.state.scaler = joblib.load(SCALER_PATH)
            logger.info(f"Loaded scaler from {SCALER_PATH}")
        else:
            app.state.scaler = None
            logger.info("No scaler found; running raw preprocessing")
    except Exception as e:
        app.state.scaler = None
        logger.error(f"Failed to load scaler: {e}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": bool(getattr(app.state, 'model_loaded', False))}


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(payload: SinglePredictionRequest, ok: bool = Depends(get_api_key)):
    if not getattr(app.state, 'model_loaded', False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = payload.features
        x = preprocess_single(features, scaler=getattr(app.state, 'scaler', None))
        model = app.state.model
        # prediction
        preds = model.predict(np.array([x]))
        probability, pred_class, class_name = _interpret_prediction(preds)
        result = {
            "prediction": int(pred_class)+3,
            "probability": float(round(probability, 6)),
            "class_name": class_name
        }
        logger.info(f"/predict - OK - input_len={len(features)} pred={pred_class} prob={probability}")
        return result
    except Exception as e:
        logger.error(f"/predict - ERROR - {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(payload: BatchPredictionRequest, ok: bool = Depends(get_api_key)):
    if not getattr(app.state, 'model_loaded', False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        data = payload.data
        X = preprocess_batch(data, scaler=getattr(app.state, 'scaler', None))
        model = app.state.model
        preds = model.predict(X)
        results = []
        for p in preds:
            probability, pred_class, class_name = _interpret_prediction(p)
            results.append({
                "prediction": int(pred_class),
                "probability": float(round(probability, 6)),
                "class_name": class_name
            })
        logger.info(f"/predict_batch - OK - n_samples={len(data)}")
        return {"predictions": results}
    except Exception as e:
        logger.error(f"/predict_batch - ERROR - {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
def model_info():
    if not getattr(app.state, 'model_loaded', False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        model = app.state.model
        info = {
            "input_shape": getattr(model, 'input_shape', None),
            "output_shape": getattr(model, 'output_shape', None),
            "layers": [layer.__class__.__name__ for layer in model.layers]
        }
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature_importance")
def feature_importance():
    # For neural networks there's no simple feature importance; user can plug SHAP
    raise HTTPException(status_code=501, detail="Feature importance not implemented. Use /explain for SHAP-based explanations if available.")


@app.post("/explain")
def explain(payload: SinglePredictionRequest):
    """Return SHAP values for a single sample. Requires shap to be installed and compatible model."""
    try:
        import shap
    except Exception:
        raise HTTPException(status_code=501, detail="shap package is not installed on the server")

    if not getattr(app.state, 'model_loaded', False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        x = preprocess_single(payload.features, scaler=getattr(app.state, 'scaler', None))
        model = app.state.model
        # This is a simple approach; for large models or custom layers consider custom explainer
        explainer = shap.Explainer(model, masker=x)
        shap_values = explainer(np.array([x]))
        return {"shap_values": shap_values.values.tolist()}
    except Exception as e:
        logger.error(f"/explain - ERROR - {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def _interpret_prediction(preds):
    """Interpret model output (binary or multiclass).
    Returns (probability, predicted_class_index, class_name)
    """
    # preds could be (1,1) sigmoid, (1,) or (n_classes,)
    arr = np.array(preds).squeeze()
    if arr.ndim == 0:
        # single value
        prob = float(arr)
        pred_class = 1 if prob >= 0.5 else 0
        class_name = "Positive" if pred_class == 1 else "Negative"
        return prob, pred_class, class_name
    elif arr.ndim == 1:
        # could be multiclass probabilities
        if len(arr) == 2:
            # binary probs for two classes
            prob = float(arr[1])
            pred_class = int(np.argmax(arr))
            class_name = f"Class_{pred_class}"
            return prob, pred_class, class_name
        else:
            pred_class = int(np.argmax(arr))
            prob = float(arr[pred_class])
            class_name = f"Class_{pred_class}"
            return prob, pred_class, class_name
    else:
        # fallback
        pred_class = int(np.argmax(arr))
        prob = float(arr.flatten()[pred_class])
        class_name = f"Class_{pred_class}"
        return prob, pred_class, class_name