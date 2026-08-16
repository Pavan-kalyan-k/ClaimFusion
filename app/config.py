import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Model Paths
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "best (1).pt"))
    KERAS_MODEL_PATH: str = os.getenv("KERAS_MODEL_PATH", str(BASE_DIR / "best_mobilenetv2.keras"))
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", str(BASE_DIR / "gradient_boosting_model.pkl"))

    # Thresholds
    YOLO_CONF_THRESHOLD: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.45

    # Directories
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    
    # Keras Settings
    KERAS_IMG_SIZE: tuple = (300, 300)
    SEVERITY_CLASSES: list = ['01-minor', '02-moderate', '03-severe']

settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
