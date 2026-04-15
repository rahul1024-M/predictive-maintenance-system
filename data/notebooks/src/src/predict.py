import joblib
import numpy as np

# Load trained model
model = joblib.load("../models/failure_model.pkl")

# Example sensor data
sample = np.array([[300, 310, 1500, 40, 10]])

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Machine Failure Likely")
else:
    print("Machine Operating Normally")
