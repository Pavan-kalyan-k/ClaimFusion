import sys
import numpy as np
import sklearn
import joblib

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Joblib:", joblib.__version__)

MODEL_PATH = "gradient_boosting_model.pkl"

print("Loading:", MODEL_PATH)

try:

    model = joblib.load(MODEL_PATH)

    print("SUCCESS")
    print("Model type:", type(model))

except Exception as e:

    import traceback

    print("FAILED")
    print("Exception type:", type(e).__name__)
    print("Exception:", str(e))

    traceback.print_exc()

    raise
