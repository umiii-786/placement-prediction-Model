import unittest
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os 

class TestModelPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model and data once for all tests"""

        # Load trained model
        with open(os.path.join('models',"gradient_boosting_model.pkl"), "rb") as f:
            cls.model = pickle.load(f)

        # Load dataset (replace with your path)
        df = pd.read_csv("data/processed/test_ds.csv")

        X = df.drop("target", axis=1)
        y = df["target"]

        # Split data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Predictions
        cls.y_train_pred = cls.model.predict(cls.X_train)
        cls.y_test_pred = cls.model.predict(cls.X_test)

        # Accuracy
        cls.train_acc = accuracy_score(cls.y_train, cls.y_train_pred)
        cls.test_acc = accuracy_score(cls.y_test, cls.y_test_pred)

    def test_model_accuracy(self):
        """Test if model meets minimum accuracy"""

        print(f"Test Accuracy: {self.test_acc}")
        self.assertGreaterEqual(
            self.test_acc, 0.75,
            "Model accuracy is too low "
        )

    def test_model_not_overfitting(self):
        """Test if model is not overfitting"""

        diff = self.train_acc - self.test_acc

        print(f"Train Accuracy: {self.train_acc}")
        print(f"Overfitting Diff: {diff}")

        self.assertLess(
            diff, 0.1,
            "Model is overfitting "
        )

    def test_train_accuracy(self):
        """Optional: Ensure model learns properly"""

        self.assertGreaterEqual(
            self.train_acc, 0.8,
            "Train accuracy too low "
        )


if __name__ == "__main__":
    unittest.main()