import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU for TensorFlow

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

# ---- KERAS COMPATIBILITY PATCHES ----
# Patch 1: Fix Keras 3 loading older Keras 3 models (input_axes error)
try:
    _orig_glorot = keras.initializers.GlorotUniform.__init__
    def patched_glorot(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        _orig_glorot(self, seed=seed)
    keras.initializers.GlorotUniform.__init__ = patched_glorot
except Exception:
    pass

# Patch 2: Fix Keras 2 loading Keras 3 models (InputLayer error)
try:
    _orig_input = keras.layers.InputLayer.__init__
    def patched_input(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        if 'optional' in kwargs:
            kwargs.pop('optional')
        _orig_input(self, *args, **kwargs)
    keras.layers.InputLayer.__init__ = patched_input
except Exception:
    pass
# -------------------------------------

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
            self.keras_model = keras.models.load_model(settings.KERAS_MODEL_PATH, compile=False)
            
        if self.ml_model is None:
            print("Loading ML model...")
            self.ml_model = joblib.load(settings.ML_MODEL_PATH)
            
        print("Models are ready.")

# Create a singleton instance
models = ModelManager()
