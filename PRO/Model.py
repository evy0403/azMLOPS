import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Argument parser for dynamic paths
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help="Path to the dataset for training")
parser.add_argument("--outputmodel", type=str, required=True, help="Directory to save the trained model")
args = parser.parse_args()

# Ensure the output folder exists
os.makedirs(args.outputmodel, exist_ok=True)

try:
    # Step 1: Load the dataset
    print(f"Loading dataset from: {args.trainingdata}")
    df = pd.read_csv(args.trainingdata)
    print("Dataset loaded successfully!")
    print(df.head())

    # Validate dataset structure
    required_columns = ['x-axis', 'y-axis', 'z-axis', 'activity']
    if not all(column in df.columns for column in required_columns):
        print(f"Error: The dataset must contain the following columns: {required_columns}")
        exit(1)

    # Extract features (x, y, z) and target (activity)
    X = df[['x-axis', 'y-axis', 'z-axis']].values
    y = df['activity'].values

    # Step 2: Split the dataset into training and testing sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train a Random Forest model
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Step 5: Save the trained model
    print(f"Saving model to: {args.outputmodel}")
    model_path = os.path.join(args.outputmodel, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved successfully at: {model_path}")

except FileNotFoundError as e:
    print(f"Error: {e}. The dataset file does not exist.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
