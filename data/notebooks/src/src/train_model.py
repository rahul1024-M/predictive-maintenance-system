import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("../data/ai4i2020.csv")

# Drop unnecessary columns
data = data.drop(columns=["UDI","Product ID","Type"])

# Target variable
y = data["Machine failure"]

# Features
X = data.drop(columns=["Machine failure"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
# Save trained model
joblib.dump(model, "../models/failure_model.pkl")

print("Model saved successfully")
