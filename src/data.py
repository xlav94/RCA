import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler


class DataDownloader:
    """
    Handles data extraction and local caching.
    Built to ensure the environment and model always have consistent,
    up-to-date market data without spamming the Yahoo Finance API.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str, data_path: str = "data/raw") -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_path = data_path
        # Ensure the storage directory exists before we try to save anything
        os.makedirs(self.data_path, exist_ok=True)

    def fetch_data(self) -> pd.DataFrame:
        """
        Smart-downloads the data. It checks the local CSV first.
        If we already have data, it only downloads the missing days (the delta) and appends it.
        This prevents us from losing historical data or re-downloading years of data unnecessarily.
        """
        filepath = os.path.join(self.data_path, 'stock_data.csv')

        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            first_date = df.index[0].date()
            last_date = df.index[-1].date()

            # If the requested start date is older than our cache, we must re-download everything
            if pd.to_datetime(self.start_date).date() < first_date:
                print("Older start date requested. Re-downloading full dataset...")
                df = yf.download(self.tickers, start=self.start_date, end=self.end_date)
                # Flatten MultiIndex columns (e.g., 'Close', 'AAPL' -> 'Close_AAPL') for a clean 2D tabular format
                df.columns = ['_'.join(col) for col in df.columns]
                df.to_csv(filepath)
                return df

            # If our cache is just slightly behind the end_date, fetch only the missing days
            if last_date < pd.to_datetime(self.end_date).date():
                new_data = yf.download(self.tickers, start=last_date + timedelta(days=1), end=self.end_date)

                if not new_data.empty:
                    new_data.columns = ['_'.join(col) for col in new_data.columns]
                    df = pd.concat([df, new_data])
                    df.to_csv(filepath)
            return df

        else:
            # First time running the script: download everything
            df = yf.download(self.tickers, start=self.start_date, end=self.end_date)
            df.columns = ['_'.join(col) for col in df.columns]
            df.to_csv(filepath)
            return df

    def get_feature_matrix(self, features: List[str] = ['Open']) -> np.ndarray:
        """
        API for the RL Environment.
        Takes the flat master CSV and slices it into a pure 2D NumPy array (tensor-ready).
        Usage for Env Dev: get_feature_matrix(['Close']) returns just the closing prices for the reward calculation.
        """
        filepath = os.path.join(self.data_path, 'stock_data.csv')
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        selected_columns = []
        for feature in features:
            for col in df.columns:
                if col.startswith(f"{feature}_"):
                    selected_columns.append(col)

        filtered_df = df[selected_columns]
        return filtered_df.values


class DataPreprocessor:
    """
    Transforms raw market data into stationary signals suitable for an LSTM Actor-Critic model.
    """

    def __init__(self, df: pd.DataFrame, tickers: List[str]) -> None:
        self.df = df.copy()
        self.tickers = tickers
        self.scaler = StandardScaler()

    def add_features(self, window: int = 20) -> pd.DataFrame:
        """
        Feature Engineering for the LSTM.
        These features provide the model with stationary context regarding trend and risk.
        """
        for ticker in self.tickers:
            close_col = f"Close_{ticker}"

            if close_col in self.df.columns:
                # 1. Returns: Converts infinitely rising prices into a stationary signal hovering around 0.
                self.df[f"Return_{ticker}"] = self.df[close_col].pct_change()

                # 2. SMA (Trend): Smooths daily noise so the LSTM can identify the macro market direction.
                self.df[f"SMA_{window}_{ticker}"] = self.df[close_col].rolling(window=window).mean()

                # 3. Volatility (Risk): Gives the Critic network a "panic meter" to help learn risk-averse policies.
                self.df[f"Vol_{window}_{ticker}"] = self.df[f"Return_{ticker}"].rolling(window=window).std()

        # Drop the initial rows (first 'window' days) that now contain NaNs due to the rolling calculations
        self.df.dropna(inplace=True)
        return self.df

    def split_train_test(self, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Slices the dataset chronologically.
        Crucial for time-series: we cannot randomly shuffle data, otherwise the agent peeks into the future.
        """
        self.df.index = pd.to_datetime(self.df.index)
        train_df = self.df[self.df.index < split_date].copy()
        test_df = self.df[self.df.index >= split_date].copy()
        return train_df, test_df

    def scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Normalizes the features so the LSTM inputs are roughly bounded between -1 and 1.
        NOTE FOR MODEL DEV: The scaler is fit ONLY on the training data to strictly prevent data leakage into the test set.
        """
        self.scaler.fit(train_df[feature_cols])
        train_df[feature_cols] = self.scaler.transform(train_df[feature_cols])
        test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])
        return train_df, test_df