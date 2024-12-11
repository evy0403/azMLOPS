import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# I dynamically providing dataset and output paths
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, default="C:\\Users\\Evelyn\\azure-ml-cicd\\production\\mydata.csv",
                    help="Path to the dataset for training")
parser.add_argument("--outputmodel", type=str, default="C:\\Users\\Evelyn\\azure-ml-cicd\\trained_model",
                    help="Directory to save the trained model")
args = parser.parse_args()

# Loading the dataset
print(f"Loading dataset from: {args.trainingdata}")
df = pd.read_csv(args.trainingdata)
print("Dataset loaded successfully!")
print(df.head())

# Extracting features (x, y, z) and target (activity)
X = df[['x-axis', 'y-axis', 'z-axis']].values
y = df['activity'].values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Saving the trained model
print(f"Saving model to: {args.outputmodel}")
os.makedirs(args.outputmodel, exist_ok=True)  # Ensure the output folder exists
model_path = os.path.join(args.outputmodel, "model.pkl")
joblib.dump(model, model_path)
print(f"Model saved successfully at: {model_path}")
