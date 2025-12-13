import os
import unittest
import numpy as np
import pandas as pd

try:
    from space_titanic import TitanicModel
except Exception as e:
    TitanicModel = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class TestTitanic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TitanicModel is None:
            raise unittest.SkipTest(f"Could not import TitanicModel: {_IMPORT_ERROR}")

    def setUp(self):
        # Prefer instance methods (works for both instance and @staticmethod in most designs)
        self.space = TitanicModel("titanic") if callable(TitanicModel) else TitanicModel

    def test_preprocess_fills_numeric_missing(self):
        df = pd.DataFrame(
            {
                "num1": [1.0, np.nan, 3.0],
                "num2": [np.nan, 5.0, 6.0],
                "cat": ["a", None, "c"],
            }
        )
        processed = self.space.preprocess(df.copy())
        self.assertFalse(processed[["num1", "num2"]].isna().any().any())
        self.assertTrue(processed["cat"].isna().any())

    def test_encode_encodes_object_columns_except_passengerid(self):
        df = pd.DataFrame(
            {
                "PassengerId": ["0001_01", "0002_01", "0003_01"],
                "HomePlanet": ["Earth", "Europa", "Mars"],
                "Cabin": ["A/0/S", "B/1/P", "C/2/S"],
            }
        )
        self.space.encode(df)
        self.assertEqual(df["PassengerId"].dtype.kind, "O")
        self.assertIn(df["HomePlanet"].dtype.kind, "biufc")
        self.assertIn(df["Cabin"].dtype.kind, "biufc")

    def test_visualize_data_creates_missing_png(self):
        out = "missing.png"
        if os.path.exists(out):
            os.remove(out)

        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        self.space.visualize_data(df)

        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)
        os.remove(out)

    def test_visualize_correlation_creates_correlation_png(self):
        out = "correlation.png"
        if os.path.exists(out):
            os.remove(out)

        df = pd.DataFrame(
            {
                "Feature1": [1, 2, 3, 4],
                "Feature2": [4, 3, 2, 1],
                "Transported": [0, 1, 0, 1],
            }
        )
        self.space.visualize_correlation(df)

        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)
        os.remove(out)

    def test_visualize_training_process_creates_training_knn_png(self):
        out = "training_knn.png"
        if os.path.exists(out):
            os.remove(out)

        scores = {1: 0.5, 2: 0.6, 3: 0.55}
        self.space.visualize_training_process(scores)

        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)
        os.remove(out)


if __name__ == "__main__":
    unittest.main()
