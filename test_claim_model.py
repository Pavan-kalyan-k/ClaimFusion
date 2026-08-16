import numpy as np
import sklearn
import joblib
import pandas as pd
from app.preprocessing import prepare_ml_features

print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Joblib:", joblib.__version__)

model = joblib.load("gradient_boosting_model.pkl")

print("CLAIM MODEL LOADED SUCCESSFULLY")
print("Model:", type(model))

features = prepare_ml_features(["bumper", "headlamp"], "moderate")
pred = model.predict(features)
print("CLAIM PREDICTION SUCCESSFUL:", pred[0])
