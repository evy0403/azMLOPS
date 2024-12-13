import unittest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse

# Argument parser for dynamic paths
def parse_args():
    parser = argparse.ArgumentParser(description="Unit Testing for Model Evaluation")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to the trained model file (.pkl)")
    parser.add_argument("--datapath", type=str, required=True, help="Path to the preprocessed dataset file (.csv)")
    return parser.parse_args()

class TestModelEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up resources for all tests."""
        args = parse_args()
        cls.model = joblib.load(args.modelpath)
        cls.data = pd.read_csv(args.datapath)

        # Validate dataset structure
        required_columns = ['x-axis', 'y-axis', 'z-axis', 'activity']
        if not all(column in cls.data.columns for column in required_columns):
            raise ValueError(f"The dataset must contain the following columns: {required_columns}")

        cls.X = cls.data[['x-axis', 'y-axis', 'z-axis']].values
        cls.y = pd.factorize(cls.data['activity'])[0]

    def test_model_accuracy(self):
        """Test if model accuracy is above a threshold."""
        y_pred = self.model.predict(self.X)
        accuracy = accuracy_score(self.y, y_pred)
        self.assertGreater(accuracy, 0.8, "Accuracy should be greater than 80%")

    def test_confusion_matrix_shape(self):
        """Test if confusion matrix dimensions are valid."""
        y_pred = self.model.predict(self.X)
        cm = confusion_matrix(self.y, y_pred)
        self.assertEqual(cm.shape[0], cm.shape[1], "Confusion matrix should be square")
        self.assertEqual(cm.shape[0], len(set(self.y)), "Confusion matrix size should match the number of classes")

    def test_model_predictions(self):
        """Test if model predictions match the input size."""
        y_pred = self.model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y), "Number of predictions should match the number of input rows")

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"])



