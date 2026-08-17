import sys
import json
import os

# Disable OneDNN to avoid AVX/memory issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Minimize TF logs and memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2

def preprocess_keras_crop(crop_img):
    crop_resized = cv2.resize(crop_img, (224, 224))
    crop_array = np.array(crop_resized, dtype=np.float32)
    crop_array = np.expand_dims(crop_array, axis=0)
    crop_array = crop_array / 255.0
    return crop_array

def run():
    try:
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        image_path = data["image_path"]
        boxes = data["boxes"]
        model_path = data["model_path"]
        
        # Import keras only in this subprocess
        import keras
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        model = keras.models.load_model(model_path, compile=False)
        
        results = []
        for box in boxes:
            x1, y1, x2, y2 = box
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                results.append(0)
                continue
            crop_batch = preprocess_keras_crop(crop)
            preds = model.predict(crop_batch, verbose=0)[0]
            severity_idx = int(np.argmax(preds))
            results.append(severity_idx)
            
        print("RESULT_JSON:" + json.dumps(results))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run()
