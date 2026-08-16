from pydantic import BaseModel
from typing import List, Optional

class DamagedPart(BaseModel):
    part: str
    confidence: float
    box: List[int]
    severity: str

class DamageDetectionResult(BaseModel):
    damaged_parts: List[DamagedPart]
    total_damaged_parts: int

class DamagePredictionResult(BaseModel):
    overall_severity: str

class ClaimPredictionResult(BaseModel):
    claim_amount: float
    prediction: str

class PredictionResponse(BaseModel):
    status: str
    summary_report: str
    damage_detection: DamageDetectionResult
    damage_prediction: DamagePredictionResult
    claim_prediction: ClaimPredictionResult
