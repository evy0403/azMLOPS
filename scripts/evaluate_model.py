import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--testdata", type=str, required=True, help="Path to the test dataset")
parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
args = parser.parse_args()

try:
    # Load test data
    print(f"Loading test data from: {args.testdata}")
    data = pd.read_csv(args.testdata)
    X_test = data[['x-axis', 'y-axis', 'z-axis']].values
    y_test = data['activity'].values

    # Load the trained model
    print(f"Loading trained model from: {args.model}")
    model = joblib.load(args.model)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

except Exception as e:
    print(f"An error occurred during evaluation: {e}")
