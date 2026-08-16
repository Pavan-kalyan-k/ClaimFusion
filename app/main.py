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

@app.get("/debug/load-claim")
def debug_load_claim():
    import joblib
    try:
        model = joblib.load("gradient_boosting_model.pkl")
        return {
            "success": True,
            "model_type": str(type(model))
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Claim model loading failed: {type(e).__name__}: {str(e)}"
        )

@app.post("/debug/predict-yolo")
async def debug_predict_yolo(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print("[DEBUG] IMAGE RECEIVED")
        print("[DEBUG] YOLO LOAD START")
        yolo_model = models.load_yolo()
        print("[DEBUG] YOLO LOAD SUCCESS")
        
        print("[DEBUG] YOLO INFERENCE START")
        results = yolo_model.predict(
            source=file_path,
            conf=settings.YOLO_CONF_THRESHOLD,
            iou=settings.YOLO_IOU_THRESHOLD,
            device="cpu",
            verbose=False
        )
        print("[DEBUG] YOLO INFERENCE SUCCESS")
        
        yolo_result = results[0]
        boxes = yolo_result.boxes
        
        raw_detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                raw_detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                })
                
        print("[DEBUG] RESPONSE CREATED")
        return {"success": True, "detections": raw_detections}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"YOLO failed: {str(e)}")
    finally:
        models.unload_yolo()
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/debug/predict-severity")
async def debug_predict_severity(file: UploadFile = File(...)):
    import cv2
    import numpy as np
    from app.preprocessing import preprocess_keras_crop
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image at {file_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        keras_model = models.load_keras()
        crop_batch = preprocess_keras_crop(image_rgb)
        preds = keras_model.predict(crop_batch, verbose=0)[0]
        severity_idx = int(np.argmax(preds))
        part_severity = settings.SEVERITY_CLASSES[severity_idx]
        
        return {"success": True, "severity": part_severity}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Severity failed: {str(e)}")
    finally:
        models.unload_keras()
        if os.path.exists(file_path):
            os.remove(file_path)

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
