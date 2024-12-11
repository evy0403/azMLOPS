import os
import pandas as pd

# Ensure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Example data cleaning logic
data = pd.read_csv("PRO/mydata.csv")
cleaned_data = data.dropna()  # Example: dropping rows with missing values
cleaned_data.to_csv("outputs/cleaned_data.csv", index=False)
