from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.models import SinglePredictionRequest, BatchPredictionRequest, PredictionResponse, BatchPredictionResponse
from src.preprocessing import preprocess_single, preprocess_batch
import numpy as np
import os
import traceback
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import joblib

app = FastAPI(
    title="Model 4 Prediction API",
    description="FastAPI app serving model_4.h5 with prediction endpoints and static frontend files",
    version="1.0.0"
)

# Mount static files at /static
# app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Serve index.html at root
# @app.get("/")
# async def serve_index():
#     return FileResponse("index.html")

# CORS - Allow Railway domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your actual domains
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid or missing API Key"
        )
    return True

# Logging
log_file = os.path.join(os.getcwd(), "prediction_requests.log")
handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger("prediction_api")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_4.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

@app.on_event("startup")
def load_model_on_startup():
    """Load the Keras model and optional scaler on app startup."""
    try:
        # Debug info
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Looking for model at: {MODEL_PATH}")
        logger.info(f"Files in base directory: {os.listdir(BASE_DIR)}")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            logger.error(f"Files in directory: {os.listdir(BASE_DIR)}")
            app.state.model = None
            app.state.model_loaded = False
            return  # Don't crash, just set model to None
        
        from tensorflow.keras.models import load_model
        app.state.model = load_model(MODEL_PATH)
        app.state.model_loaded = True
        app.state.expected_features = app.state.model.input_shape[-1]
        
        logger.info(f"✓ Loaded model from {MODEL_PATH}")
        logger.info(f"✓ Expected input features: {app.state.expected_features}")
    except Exception as e:
        app.state.model = None
        app.state.model_loaded = False
        app.state.expected_features = None
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        # Don't raise - let app start anyway

    # Load scaler
    try:
        if os.path.exists(SCALER_PATH):
            app.state.scaler = joblib.load(SCALER_PATH)
            logger.info(f"✓ Loaded scaler from {SCALER_PATH}")
        else:
            app.state.scaler = None
            logger.info(f"No scaler found at {SCALER_PATH}; running raw preprocessing")
    except Exception as e:
        app.state.scaler = None
        logger.error(f"Failed to load scaler: {e}")
    
    # Initialize SHAP
    try:
        import shap
        app.state.shap_explainer = None
        app.state.shap_available = True
        logger.info("SHAP is available")
    except ImportError:
        app.state.shap_available = False
        logger.info("SHAP not available")

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Model 4 Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "model_info": "/model_info",
            "explain": "/explain",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint with detailed status."""
    return {
        "status": "healthy",
        "model_loaded": bool(getattr(app.state, 'model_loaded', False)),
        "scaler_loaded": getattr(app.state, 'scaler', None) is not None,
        "expected_features": getattr(app.state, 'expected_features', None)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    payload: SinglePredictionRequest, 
    ok: bool = Depends(get_api_key)
):
    """Make a single prediction with input validation."""
    if not getattr(app.state, 'model_loaded', False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = payload.features
        
        # Validate input length
        expected = app.state.expected_features
        if expected and len(features) != expected:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected} features, got {len(features)}"
            )
        
        # Preprocess
        x = preprocess_single(features, scaler=getattr(app.state, 'scaler', None))
        
        # Validate for NaN/Inf after preprocessing
        if not np.all(np.isfinite(x)):
            raise HTTPException(
                status_code=400,
                detail="Invalid feature values (NaN or Inf detected after preprocessing)"
            )
        
        # Prediction
        model = app.state.model
        preds = model.predict(np.array([x]), verbose=0)
        probability, pred_class, class_name = _interpret_prediction(preds)
        
        result = {
            "prediction": int(pred_class)+3,
            "probability": float(round(probability, 6)),
            "class_name": class_name
        }
        
        logger.info(
            f"/predict - OK - input_len={len(features)} "
            f"pred={pred_class} prob={probability:.4f}"
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/predict - ERROR - {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictionRequest, 
    ok: bool = Depends(get_api_key)
):
    """Make batch predictions with validation."""
    if not getattr(app.state, 'model_loaded', False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = payload.data
        
        if not data:
            raise HTTPException(status_code=400, detail="Empty data array")
        
        # Validate all inputs have same length
        expected = app.state.expected_features
        for idx, row in enumerate(data):
            if expected and len(row) != expected:
                raise HTTPException(
                    status_code=400,
                    detail=f"Row {idx}: Expected {expected} features, got {len(row)}"
                )
        
        # Preprocess
        X = preprocess_batch(data, scaler=getattr(app.state, 'scaler', None))
        
        # Validate for NaN/Inf
        if not np.all(np.isfinite(X)):
            raise HTTPException(
                status_code=400,
                detail="Invalid feature values (NaN or Inf detected)"
            )
        
        # Predictions
        model = app.state.model
        preds = model.predict(X, verbose=0)
        
        results = []
        for p in preds:
            probability, pred_class, class_name = _interpret_prediction(p)
            results.append(
                PredictionResponse(
                    prediction=int(pred_class)+3,
                    probability=float(round(probability, 6)),
                    class_name=class_name
                )
            )
        
        logger.info(f"/predict_batch - OK - n_samples={len(data)}")
        return BatchPredictionResponse(predictions=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/predict_batch - ERROR - {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
def model_info():
    """Get detailed model information."""
    if not getattr(app.state, 'model_loaded', False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model = app.state.model
        
        # Get layer information
        layers_info = []
        for layer in model.layers:
            layers_info.append({
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(layer.output_shape),
                "params": layer.count_params()
            })
        
        info = {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "total_params": model.count_params(),
            "trainable_params": sum([layer.count_params() for layer in model.layers if layer.trainable]),
            "layers": layers_info,
            "expected_features": getattr(app.state, 'expected_features', None)
        }
        return info
        
    except Exception as e:
        logger.error(f"/model_info - ERROR - {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature_importance")
def feature_importance():
    """Feature importance endpoint (not applicable for neural networks)."""
    raise HTTPException(
        status_code=501, 
        detail="Feature importance not directly available for neural networks. "
               "Use /explain endpoint for SHAP-based explanations."
    )

@app.post("/explain")
def explain(payload: SinglePredictionRequest):
    """
    Return SHAP values for a single sample.
    Note: First call may be slow as it initializes the explainer.
    """
    if not getattr(app.state, 'shap_available', False):
        raise HTTPException(
            status_code=501, 
            detail="SHAP package is not installed on the server"
        )

    if not getattr(app.state, 'model_loaded', False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import shap
        
        # Preprocess input
        x = preprocess_single(
            payload.features, 
            scaler=getattr(app.state, 'scaler', None)
        )
        x_array = np.array([x])
        
        # Initialize explainer if not already done (lazy initialization)
        if app.state.shap_explainer is None:
            logger.info("Initializing SHAP explainer (first call)...")
            model = app.state.model
            # Use a small background dataset (ideally load from saved file)
            background = x_array  # In production, use proper background data
            app.state.shap_explainer = shap.DeepExplainer(model, background)
            logger.info("SHAP explainer initialized")
        
        # Calculate SHAP values
        shap_values = app.state.shap_explainer.shap_values(x_array)
        
        # Format response
        if isinstance(shap_values, list):
            # Multi-class output
            formatted_shap = [vals.tolist() for vals in shap_values]
        else:
            formatted_shap = shap_values.tolist()
        
        return {
            "shap_values": formatted_shap,
            "base_value": float(app.state.shap_explainer.expected_value) if hasattr(app.state.shap_explainer, 'expected_value') else None,
            "features": payload.features
        }
        
    except Exception as e:
        logger.error(f"/explain - ERROR - {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def _interpret_prediction(preds):
    """
    Interpret model output (binary or multiclass).
    Returns (probability, predicted_class_index, class_name)
    """
    arr = np.array(preds).squeeze()
    
    if arr.ndim == 0:
        # Single sigmoid output (binary classification)
        prob = float(arr)
        pred_class = 1 if prob >= 0.5 else 0
        class_name = f"Class_{pred_class+3}"
        return prob, pred_class, class_name
        
    elif arr.ndim == 1:
        # Multiclass probabilities
        pred_class = int(np.argmax(arr))
        prob = float(arr[pred_class])
        class_name = f"Class_{pred_class+3}"
        return prob, pred_class, class_name
    else:
        # Fallback for unexpected shapes
        arr_flat = arr.flatten()
        pred_class = int(np.argmax(arr_flat))
        prob = float(arr_flat[pred_class])
        class_name = f"Class_{pred_class+3}"
        return prob, pred_class, class_name

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)