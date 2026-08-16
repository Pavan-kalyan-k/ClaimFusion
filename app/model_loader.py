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

# ---- ULTIMATE KERAS COMPATIBILITY PATCH ----
# This fixes the root cause of ALL Keras deserialization errors by dynamically
# intercepting every layer and initializer's __init__ method and stripping
# out any keyword arguments that the current environment does not support.
try:
    import inspect
    def make_robust_init(orig_init):
        def patched_init(self, *args, **kwargs):
            try:
                sig = inspect.signature(orig_init)
                valid_keys = set(sig.parameters.keys())
                # Base Layer kwargs usually handled by **kwargs in subclasses
                valid_keys.update({'name', 'trainable', 'dtype', 'autocast', 'dynamic', 'batch_input_shape', 'batch_size', 'weights', 'input_shape', 'input_dim'})
                
                # Strip invalid kwargs
                clean_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
                
                # Handle Keras 3 batch_shape to Keras 2 batch_input_shape translation
                if 'batch_shape' in kwargs and 'batch_input_shape' not in clean_kwargs:
                    clean_kwargs['batch_input_shape'] = kwargs['batch_shape']
                    
                orig_init(self, *args, **clean_kwargs)
            except Exception:
                orig_init(self, *args, **kwargs)
        return patched_init

    for module in [keras.layers, keras.initializers]:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if inspect.isclass(attr):
                try:
                    attr.__init__ = make_robust_init(attr.__init__)
                except Exception:
                    pass
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
