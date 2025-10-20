from pydantic import BaseModel
from typing import List

class SinglePredictionRequest(BaseModel):
    features: List[float]

class BatchPredictionRequest(BaseModel):
    data: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    class_name: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]