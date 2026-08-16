import os
import cv2
import numpy as np

os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from app.model_loader import models
from app.preprocessing import preprocess_keras_crop, prepare_ml_features
from app.config import settings

def test_debug():
    print("==================================================")
    print("MODEL DEBUG SCRIPT")
    print("==================================================")
    
    # 1. YOLO
    try:
        yolo = models.load_yolo()
        print("YOLO LOAD: PASS")
    except Exception as e:
        print(f"YOLO LOAD: FAIL\nYOLO ERROR: {str(e)}")
        return
        
    try:
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        dummy_path = "dummy_debug.jpg"
        cv2.imwrite(dummy_path, dummy_img)
        yolo.predict(source=dummy_path, device="cpu", verbose=False)
        print("YOLO INFERENCE: PASS")
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
    except Exception as e:
        print(f"YOLO INFERENCE: FAIL\nYOLO ERROR: {str(e)}")
        return
    finally:
        models.unload_yolo()

    # 2. KERAS
    try:
        keras_model = models.load_keras()
        print("SEVERITY LOAD: PASS")
    except Exception as e:
        print(f"SEVERITY LOAD: FAIL\nSEVERITY ERROR: {str(e)}")
        return
        
    try:
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        batch = preprocess_keras_crop(dummy_crop)
        keras_model.predict(batch, verbose=0)
        print("SEVERITY INFERENCE: PASS")
    except Exception as e:
        print(f"SEVERITY INFERENCE: FAIL\nSEVERITY ERROR: {str(e)}")
        return
    finally:
        models.unload_keras()

    # 3. CLAIM
    try:
        ml_model = models.load_ml()
        print("CLAIM LOAD: PASS")
    except Exception as e:
        print(f"CLAIM LOAD: FAIL\nCLAIM ERROR: {str(e)}")
        return
        
    try:
        features = prepare_ml_features(["bumper", "headlamp"], "02-moderate")
        ml_model.predict(features)
        print("CLAIM INFERENCE: PASS")
    except Exception as e:
        print(f"CLAIM INFERENCE: FAIL\nCLAIM ERROR: {str(e)}")
        return
    finally:
        models.unload_ml()

if __name__ == "__main__":
    test_debug()
