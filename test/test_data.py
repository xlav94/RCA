import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.data import DataDownloader, DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        """Creates fake stock data to test the math without needing the internet."""
        # 100 days of fake data
        self.dates = pd.date_range(start="2023-01-01", periods=100)

        # fake prices that slowly go up to simulate a trend
        data = {
            'Open_AAPL': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            'Open_MSFT': np.linspace(200, 250, 100) + np.random.normal(0, 1, 100),
        }
        self.df = pd.DataFrame(data, index=self.dates)
        self.tickers = ["AAPL", "MSFT"]

        # initialize the preprocessor
        self.preprocessor = DataPreprocessor(self.df, self.tickers)

    def test_add_features(self):
        """Verifies that technical indicators are calculated and NaNs are dropped."""
        window = 20
        processed_df = self.preprocessor.add_features(window=window)

        # check that the new columns were created
        self.assertIn("Return_AAPL", processed_df.columns)
        self.assertIn("SMA_20_MSFT", processed_df.columns)
        self.assertIn("Vol_20_AAPL", processed_df.columns)

        # pct_change() consumes 1 day, then rolling(20) needs 20 days of returns.
        # total dropped days = window (20). 100 original - 20 dropped = 80 remaining.
        self.assertEqual(len(processed_df), 100 - window)

    def test_get_env_ready_data(self):
        """Verifies the environment gets strictly the requested columns."""
        env_df = self.preprocessor.get_env_ready_data(feature='Open')
        # The environment needs exactly 2 columns for the dot product to work
        self.assertEqual(len(env_df.columns), 2)
        self.assertIn("Open_AAPL", env_df.columns)
        self.assertIn("Open_MSFT", env_df.columns)

    def test_get_env_ready_data_missing_column(self):
        """Verifies it crashes safely if we ask for a column that doesn't exist."""
        # We don't have 'Close' prices in our dummy data, so this should raise an error
        with self.assertRaises(ValueError):
            self.preprocessor.get_env_ready_data(feature='Close')

    def test_split_train_test(self):
        """Verifies the chronological split ratio."""
        train_df, test_df = self.preprocessor.split_train_test(self.preprocessor.df, train_ratio=0.8)
        # 80% of 100 rows is 80 rows
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)

        # chronological check: Train dates must be strictly before Test dates
        self.assertTrue(train_df.index[-1] < test_df.index[0])



class TestDataDownloader(unittest.TestCase):

    @patch("pandas.read_csv")
    def test_get_feature_matrix_logic(self, mock_read_csv):
        """
        Uses a mock to intercept the CSV reading process.
        Tests if it correctly filters the flat columns into a numpy array.
        """
        # create a fake dataframe that 'read_csv' will return
        fake_data = {
            'Open_AAPL': [150, 151],
            'Close_AAPL': [152, 150],
            'Open_MSFT': [250, 252]
        }
        mock_read_csv.return_value = pd.DataFrame(fake_data)

        downloader = DataDownloader(tickers=["AAPL", "MSFT"], start_date="2020-01-01", end_date="2021-01-01")
        matrix = downloader.get_feature_matrix(features=['Open'])

        # should grab 'Open_AAPL' and 'Open_MSFT', ignoring 'Close_AAPL'
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix[0][0], 150)  # Open_AAPL day 1
        self.assertEqual(matrix[0][1], 250)  # Open_MSFT day 1