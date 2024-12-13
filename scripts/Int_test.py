import argparse
import unittest
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

# Argument parser for dynamic paths
def parse_args():
    parser = argparse.ArgumentParser(description="Integration Testing for Model Pipeline")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to save/load the trained model file (.pkl)")
    parser.add_argument("--datapath", type=str, required=True, help="Path to the preprocessed dataset file (.csv)")
    return parser.parse_args()

class TestModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up resources for all tests."""
        args = parse_args()
        cls.model_path = args.modelpath
        cls.data_path = args.datapath

        # Load the dataset
        cls.data = pd.read_csv(cls.data_path)

        # Validate dataset structure
        required_columns = ['x-axis', 'y-axis', 'z-axis', 'activity']
        if not all(column in cls.data.columns for column in required_columns):
            raise ValueError(f"The dataset must contain the following columns: {required_columns}")

        cls.X = cls.data[['x-axis', 'y-axis', 'z-axis']].values
        cls.y = pd.factorize(cls.data['activity'])[0]

    def test_pipeline_training_and_saving(self):
        """Test training a model and saving it to disk."""
        # Train a new model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X, self.y)

        # Save the model
        joblib.dump(model, self.model_path)
        self.assertTrue(os.path.exists(self.model_path), "Model file should be saved to disk")

    def test_pipeline_loading_and_evaluation(self):
        """Test loading the model and evaluating its accuracy."""
        # Load the model
        model = joblib.load(self.model_path)

        # Make predictions
        y_pred = model.predict(self.X)

        # Evaluate metrics
        accuracy = accuracy_score(self.y, y_pred)
        f1 = f1_score(self.y, y_pred, average='weighted')

        # Assert metrics
        self.assertGreater(accuracy, 0.8, "Accuracy should be greater than 80%")
        self.assertGreater(f1, 0.8, "F1 score should be greater than 80%")

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"])
