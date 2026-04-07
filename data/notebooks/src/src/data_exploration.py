import pandas as pd

# Load dataset
data = pd.read_csv("../data/ai4i2020.csv")

# Show first rows
print("First 5 rows of dataset:")
print(data.head())

# Show dataset information
print("\nDataset Info:")
print(data.info())

# Show statistics
print("\nDataset Statistics:")
print(data.describe())
