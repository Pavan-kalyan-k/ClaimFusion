import joblib
import torch
# Monkey patch torch.load to bypass weights_only in PyTorch 2.6 for Ultralytics
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO
import keras
from app.config import settings

class ModelManager:
    def __init__(self):
        self.yolo_model = None
        self.keras_model = None
        self.ml_model = None

    def load_models(self):
        if self.yolo_model is None:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
            
        if self.keras_model is None:
            print("Loading Keras model...")
            self.keras_model = keras.models.load_model(settings.KERAS_MODEL_PATH)
            
        if self.ml_model is None:
            print("Loading ML model...")
            self.ml_model = joblib.load(settings.ML_MODEL_PATH)
            
        print("Models are ready.")

# Create a singleton instance
models = ModelManager()
