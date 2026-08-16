import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import pickle

print("Retraining claim model with CURRENT environment...")

# 1. Load Data
df = pd.read_csv("car_insurance_claims_dataset.csv")

# 2. Preprocess
df['severity'] = df['severity'].map({'medium':'moderate','high':'severe','low':'minor'})
df.drop(['claim_id','damage_score','estimated_repair_cost'], axis=1, inplace=True)

x = df.drop("insurance_payout", axis=1)
y = df["insurance_payout"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Encode Severity
le = LabelEncoder()
x_train['severity'] = le.fit_transform(x_train['severity'])
x_test['severity'] = le.transform(x_test['severity'])

print("Severity classes:", le.classes_)

# 4. Train Gradient Boosting
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(x_train, y_train)
print("Train Accuracy:", gbr.score(x_train, y_train))
print("Test Accuracy:", gbr.score(x_test, y_test))

# 5. Save Model
filename = 'gradient_boosting_model.pkl'
import joblib
joblib.dump(gbr, filename)
print(f"Model saved to {filename}")
