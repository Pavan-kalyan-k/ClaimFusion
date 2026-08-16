import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

model = keras.models.load_model(
    "best_mobilenetv2.keras",
    compile=False
)

print("Model loaded successfully")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

dummy = np.zeros(
    (1, 300, 300, 3),
    dtype=np.float32
)

prediction = model.predict(dummy, verbose=0)

print("Prediction successful")
print("Prediction shape:", prediction.shape)
print("Prediction:", prediction)
