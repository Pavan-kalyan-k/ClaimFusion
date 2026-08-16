from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil

from app.model_loader import models
from app.config import settings
from app.schemas import PredictionResponse
from app.pipeline import run_prediction_pipeline

app = FastAPI(
    title="Car Damage Assessment API",
    description="API for detecting car damage, assessing severity, and estimating claim payouts using YOLO, MobileNetV2, and Gradient Boosting.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://claimfusion-ai.onrender.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "service": "ClaimFusion 360"
    }

@app.get("/model-status")
def model_status():
    def get_status(model_prop):
        return "loaded" if model_prop is not None else "not_loaded"
    
    return {
        "yolo": get_status(models.yolo_model),
        "severity": get_status(models.keras_model),
        "claim": get_status(models.ml_model)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_claim(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    # Save the file temporarily
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run Pipeline
        response = run_prediction_pipeline(file_path)
        return response
        
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

# Mount the static directory to serve the frontend UI (Must be at the bottom)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
