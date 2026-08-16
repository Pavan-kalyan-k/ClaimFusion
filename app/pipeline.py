import cv2
import numpy as np

from app.model_loader import models
from app.config import settings
from app.preprocessing import (
    remove_duplicate_boxes, 
    preprocess_keras_crop, 
    prepare_ml_features
)
from app.schemas import (
    PredictionResponse,
    DamageDetectionResult,
    DamagedPart,
    DamagePredictionResult,
    ClaimPredictionResult
)

def run_prediction_pipeline(image_path: str) -> PredictionResponse:
    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. YOLO Inference
    yolo_model = models.load_yolo()
    results = yolo_model.predict(
        source=image_path,
        conf=settings.YOLO_CONF_THRESHOLD,
        iou=settings.YOLO_IOU_THRESHOLD,
        device="cpu",
        verbose=False
    )
    
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
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })
            
    models.unload_yolo() # Free YOLO memory!
            
    # 3. Process Detections and Keras Inference
    final_detections = remove_duplicate_boxes(raw_detections)
    
    damaged_parts = []
    detected_part_names = set()
    severities = []
    
    if len(final_detections) > 0:
        keras_model = models.load_keras()
        for det in final_detections:
            x1, y1, x2, y2 = det["box"]
            
            # Extract Crop
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # Keras Prediction
            crop_batch = preprocess_keras_crop(crop)
            preds = keras_model.predict(crop_batch, verbose=0)[0]
            severity_idx = int(np.argmax(preds))
            part_severity = settings.SEVERITY_CLASSES[severity_idx]
            
            severities.append(severity_idx) # Keep integer for max calculation
            
            damaged_parts.append(DamagedPart(
                part=det["class_name"],
                confidence=det["confidence"],
                box=det["box"],
                severity=part_severity
            ))
            
            detected_part_names.add(det["class_name"])
            
        models.unload_keras() # Free Keras memory!
            
    # 4. Determine Overall Severity
    if not severities:
        overall_severity_str = "No Damage"
    else:
        # 0: minor, 1: moderate, 2: severe
        max_severity_idx = max(severities)
        overall_severity_str = settings.SEVERITY_CLASSES[max_severity_idx]
        
    # 5. ML Feature Engineering & Prediction
    if overall_severity_str == "No Damage":
        claim_amount = 0.0
    else:
        ml_model = models.load_ml()
        ml_features = prepare_ml_features(list(detected_part_names), overall_severity_str)
        claim_amount = float(ml_model.predict(ml_features)[0])
        models.unload_ml() # Free ML memory!
        
    # 6. Build Response
    if overall_severity_str == "No Damage":
        summary_report = "No vehicle damage was detected in the image. No claim is required."
    else:
        parts_str = ", ".join(list(detected_part_names))
        summary_report = f"We detected {len(detected_part_names)} damaged vehicle part(s): {parts_str}. The overall severity is {overall_severity_str.upper()}, resulting in an estimated insurance claim payout of ${claim_amount:,.2f}."

    return PredictionResponse(
        status="success",
        summary_report=summary_report,
        damage_detection=DamageDetectionResult(
            damaged_parts=damaged_parts,
            total_damaged_parts=len(detected_part_names)
        ),
        damage_prediction=DamagePredictionResult(
            overall_severity=overall_severity_str
        ),
        claim_prediction=ClaimPredictionResult(
            claim_amount=round(claim_amount, 2),
            prediction="Claim approved" if claim_amount > 0 else "No claim required"
        )
    )
