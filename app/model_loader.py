import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU for TensorFlow

import joblib
import torch
import numpy as np

# ---- ULTIMATE NUMPY COMPATIBILITY PATCH ----
# This fixes the numpy 2.x to numpy 1.x BitGenerator/MT19937 pickle crash
# by intercepting the random state reconstruction and ignoring the state validation.
try:
    from numpy.random import _pickle
    
    class FakeRandomState(np.random.RandomState):
        def __setstate__(self, state):
            pass
            
    _orig_randomstate_ctor = _pickle.__randomstate_ctor
    def patched_randomstate_ctor(bit_generator_name="MT19937"):
        return FakeRandomState()
    _pickle.__randomstate_ctor = patched_randomstate_ctor
    
    class FakeBitGenerator:
        def __setstate__(self, state):
            pass
            
    _orig_bit_generator_ctor = _pickle.__bit_generator_ctor
    def patched_bit_generator_ctor(bit_generator_name="MT19937"):
        return FakeBitGenerator()
    _pickle.__bit_generator_ctor = patched_bit_generator_ctor
except Exception:
    pass
# ---------------------------------------------

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

import gc

class ModelManager:
    def __init__(self):
        self.yolo_model = None
        self.keras_model = None
        self.ml_model = None

    def load_yolo(self):
        if self.yolo_model is None:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
        return self.yolo_model
        
    def unload_yolo(self):
        self.yolo_model = None
        gc.collect()

    def load_keras(self):
        if self.keras_model is None:
            print("Loading Keras model...")
            self.keras_model = keras.models.load_model(settings.KERAS_MODEL_PATH, compile=False)
        return self.keras_model
        
    def unload_keras(self):
        self.keras_model = None
        keras.backend.clear_session()
        gc.collect()

    def load_ml(self):
        if self.ml_model is None:
            print("Loading ML model...")
            self.ml_model = joblib.load(settings.ML_MODEL_PATH)
        return self.ml_model
        
    def unload_ml(self):
        self.ml_model = None
        gc.collect()

# Create a singleton instance
models = ModelManager()
