from pydantic import BaseModel, conlist
from typing import List, Any, Dict

# Assuming your model expects numeric features
class SinglePredictionRequest(BaseModel):
    # features array of floats. Use conlist to enforce length if you know it e.g. conlist(float, min_items=11, max_items=11)
    features: List[float]

class BatchPredictionRequest(BaseModel):
    data: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    class_name: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]