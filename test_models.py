import os
import cv2
import numpy as np

# Force single threading BEFORE imports just like in production
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from app.model_loader import models
from app.preprocessing import preprocess_keras_crop, prepare_ml_features
from app.config import settings

def test_yolo():
    print("\n--- Testing YOLO ---")
    try:
        yolo = models.load_yolo()
        print("YOLO loaded successfully.")
        
        # Test inference with dummy image
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        dummy_path = "dummy_test.jpg"
        cv2.imwrite(dummy_path, dummy_img)
        
        results = yolo.predict(source=dummy_path, device="cpu", verbose=False)
        print("YOLO inference successful.")
        
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            
    except Exception as e:
        print(f"YOLO FAILED: {str(e)}")
        raise
    finally:
        models.unload_yolo()

def test_keras():
    print("\n--- Testing Keras MobileNetV2 ---")
    try:
        keras_model = models.load_keras()
        print("Keras loaded successfully.")
        
        # Test inference with dummy crop
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        batch = preprocess_keras_crop(dummy_crop)
        preds = keras_model.predict(batch, verbose=0)
        print(f"Keras inference successful. Prediction shape: {preds.shape}")
        
    except Exception as e:
        print(f"KERAS FAILED: {str(e)}")
        raise
    finally:
        models.unload_keras()

def test_ml():
    print("\n--- Testing Scikit-Learn Gradient Boosting ---")
    try:
        ml_model = models.load_ml()
        print("ML model loaded successfully.")
        
        # Test inference with dummy features
        features = prepare_ml_features(["bumper", "headlamp"], "02-moderate")
        pred = ml_model.predict(features)
        print(f"ML inference successful. Prediction: {pred}")
        
    except Exception as e:
        print(f"ML MODEL FAILED: {str(e)}")
        raise
    finally:
        models.unload_ml()

if __name__ == "__main__":
    print("Starting Model Tests...")
    test_yolo()
    test_keras()
    test_ml()
    print("\nAll tests passed!")
