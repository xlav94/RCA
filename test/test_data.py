import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data import DataPipeline


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.tickers = ["AAPL", "MSFT"]
        self.pipeline = DataPipeline(
            tickers=self.tickers,
            start_date="2023-01-01",
            end_date="2024-01-01"
        )

        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100)
        self.fake_raw = pd.DataFrame({
            'Open_AAPL': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            'Open_MSFT': np.linspace(200, 250, 100) + np.random.normal(0, 1, 100),
            'Close_AAPL': np.linspace(101, 151, 100),
            'Close_MSFT': np.linspace(201, 251, 100),
        }, index=dates)

    @patch.object(DataPipeline, '_get_raw')
    def test_get_env_data_returns_correct_columns(self, mock_get_raw):
        mock_get_raw.return_value = self.fake_raw
        env_df = self.pipeline.get_env_data(feature='Open')

        self.assertEqual(len(env_df.columns), 2)
        self.assertIn("Open_AAPL", env_df.columns)
        self.assertIn("Open_MSFT", env_df.columns)

    @patch.object(DataPipeline, '_get_raw')
    def test_get_env_data_missing_column_raises(self, mock_get_raw):
        mock_get_raw.return_value = self.fake_raw
        with self.assertRaises(ValueError):
            self.pipeline.get_env_data(feature='High')

    @patch.object(DataPipeline, '_get_raw')
    def test_get_env_data_column_order_matches_tickers(self, mock_get_raw):
        mock_get_raw.return_value = self.fake_raw
        env_df = self.pipeline.get_env_data(feature='Open')
        expected_cols = [f"Open_{t}" for t in self.tickers]
        self.assertEqual(list(env_df.columns), expected_cols)


if __name__ == "__main__":
    unittest.main()