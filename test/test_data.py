import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data import DataPipeline, DataDownloader


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

class TestDataDownloader(unittest.TestCase):
    def setUp(self):
        self.tickers = ["AAPL", "MSFT"]
        self.downloader = DataDownloader(
            tickers=self.tickers,
            start_date="2023-01-01",
            end_date="2024-01-01"
        )

    @patch("src.data.yf.download")
    @patch("src.data.os.path.exists")
    def test_fetch_data_downloads_when_no_cache(self, mock_exists, mock_download):
        mock_exists.return_value = False

        dates = pd.date_range(start="2023-01-01", periods=10)
        arrays = [
            ["Open", "Open", "Close", "Close"],
            ["AAPL", "MSFT", "AAPL", "MSFT"]
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        fake_df = pd.DataFrame(
            np.random.rand(10, 4),
            index=dates,
            columns=index
        )
        mock_download.return_value = fake_df

        with patch("builtins.open", unittest.mock.mock_open()):
            with patch("pandas.DataFrame.to_csv"):
                result = self.downloader.fetch_data()

        mock_download.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)

    @patch("src.data.pd.read_csv")
    @patch("src.data.os.path.exists")
    def test_fetch_data_uses_cache_when_up_to_date(self, mock_exists, mock_read_csv):
        mock_exists.return_value = True

        dates = pd.date_range(start="2023-01-01", periods=10)
        fake_cached = pd.DataFrame({
            'Open_AAPL': np.linspace(100, 110, 10),
            'Open_MSFT': np.linspace(200, 210, 10),
        }, index=dates)
        mock_read_csv.return_value = fake_cached

        with patch("src.data.yf.download") as mock_download:
            self.downloader.end_date = "2023-01-10"
            self.downloader.fetch_data()
            mock_download.assert_not_called()

class TestChronologicalSplit(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100)
        self.df = pd.DataFrame({
            'Open_AAPL': np.linspace(100, 150, 100),
            'Open_MSFT': np.linspace(200, 250, 100),
        }, index=dates)

    def test_split_ratio(self):
        train_size = int(len(self.df) * 0.8)
        df_train = self.df.iloc[:train_size]
        df_test = self.df.iloc[train_size:]

        self.assertEqual(len(df_train), 80)
        self.assertEqual(len(df_test), 20)

    def test_split_no_overlap(self):
        train_size = int(len(self.df) * 0.8)
        df_train = self.df.iloc[:train_size]
        df_test = self.df.iloc[train_size:]

        overlap = df_train.index.intersection(df_test.index)
        self.assertEqual(len(overlap), 0)

    def test_split_chronological_order(self):
        train_size = int(len(self.df) * 0.8)
        df_train = self.df.iloc[:train_size]
        df_test = self.df.iloc[train_size:]

        self.assertTrue(df_train.index[-1] < df_test.index[0])

if __name__ == "__main__":
    unittest.main()