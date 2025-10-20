# FastAPI Model API (model_4.h5)

## Overview
This backend serves `model_4.h5` through a FastAPI app. It provides endpoints for single and batch predictions, health check, model info, and a basic SHAP explanation endpoint.

## Files
- `main.py` - FastAPI app (startup loads the model)
- `models.py` - Pydantic request/response models
- `preprocessing.py` - Preprocessing functions (load scaler if `scaler.joblib` exists)
- `requirements.txt` - Python dependencies

## Setup
1. Create a Python 3.9+ virtual environment and activate it.
```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt