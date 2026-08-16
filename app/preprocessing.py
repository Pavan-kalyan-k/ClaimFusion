import cv2
import numpy as np
import keras
import pandas as pd

from app.config import settings

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area
    if union_area == 0: return 0
    return intersection_area / union_area

def remove_duplicate_boxes(detections, iou_threshold=settings.YOLO_IOU_THRESHOLD):
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    final_detections = []
    
    for det in detections:
        keep = True
        for sel in final_detections:
            if det["class_id"] == sel["class_id"]:
                iou = calculate_iou(det["box"], sel["box"])
                if iou > iou_threshold:
                    keep = False
                    break
        if keep:
            final_detections.append(det)
            
    return final_detections

def preprocess_keras_crop(crop_rgb):
    resized_crop = cv2.resize(crop_rgb, settings.KERAS_IMG_SIZE)
    crop_array = keras.utils.img_to_array(resized_crop) / 255.0
    crop_batch = np.expand_dims(crop_array, axis=0)
    return crop_batch

def prepare_ml_features(damaged_parts_list, overall_severity):
    # 1. Binary parts vector mapping
    part_mapping = {
        "Door": "door_damaged",
        "Bonnet": "bonnet_damaged",
        "Bumper": "bumper_damaged",
        "Dickey": "dickey_damaged",
        "Fender": "fender_damaged",
        "Light": "light_damaged",
        "Windshield": "windshield_damaged"
    }
    
    features = {
        "door_damaged": 0,
        "bonnet_damaged": 0,
        "bumper_damaged": 0,
        "dickey_damaged": 0,
        "fender_damaged": 0,
        "light_damaged": 0,
        "windshield_damaged": 0
    }
    
    for part in damaged_parts_list:
        col_name = part_mapping.get(part)
        if col_name:
            features[col_name] = 1
            
    # 2. Total parts damaged
    total_parts = sum(features.values())
    
    # 3. Encode Severity
    # Keras predicts: 01-minor, 02-moderate, 03-severe
    # ML expects label encoded integer: 0: minor, 1: moderate, 2: severe
    severity_mapping = {
        "01-minor": 0,
        "02-moderate": 1,
        "03-severe": 2,
        "No Damage": 0  # fallback
    }
    encoded_severity = severity_mapping.get(overall_severity, 0)
    
    features["severity"] = encoded_severity
    features["total_parts_damaged"] = total_parts
    
    # Ensure correct column order as expected by ML model
    columns = [
        "door_damaged", "bonnet_damaged", "bumper_damaged", 
        "dickey_damaged", "fender_damaged", "light_damaged", 
        "windshield_damaged", "severity", "total_parts_damaged"
    ]
    
    df = pd.DataFrame([features], columns=columns)
    return df
