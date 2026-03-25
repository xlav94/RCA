import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
import pywt


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
            open_col = f"Open_{ticker}"

            if open_col in self.df.columns:
                # 1. Returns: Converts infinitely rising prices into a stationary signal hovering around 0.
                self.df[f"Return_{ticker}"] = self.df[open_col].pct_change()

                # 2. SMA (Trend): Smooths daily noise so the LSTM can identify the macro market direction.
                self.df[f"SMA_{window}_{ticker}"] = self.df[open_col].rolling(window=window).mean()

                # 3. Volatility (Risk): Gives the Critic network a "panic meter" to help learn risk-averse policies.
                self.df[f"Vol_{window}_{ticker}"] = self.df[f"Return_{ticker}"].rolling(window=window).std()

        # Drop the initial rows (first 'window' days) that now contain NaNs due to the rolling calculations
        self.df.dropna(inplace=True)
        return self.df

    def split_train_test(self, input_df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Slices the dataset chronologically using a percentage.
        E.g., train_ratio=0.8 means 80% of the data is for training, 20% for testing.
        """
        # Ensure chronological order before splitting
        input_df = input_df.sort_index()
        split_idx = int(len(input_df) * train_ratio)
        train_df = input_df.iloc[:split_idx].copy()
        test_df = input_df.iloc[split_idx:].copy()

        return train_df, test_df

    def get_env_ready_data(self, feature: str = 'Open') -> pd.DataFrame:
        """
        Extracts ONLY the requested feature for the assets, strictly ordered.
        This prevents the np.dot shape mismatch in the environment's reward calculation.
        """
        # Create a list of the exact column names the env needs (e.g., ['Open_AAPL', 'Open_MSFT'])
        ordered_cols = [f"{feature}_{ticker}" for ticker in self.tickers]

        # Verify columns exist to prevent silent typos
        for col in ordered_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Slice the dataframe to just those columns
        env_df = self.df[ordered_cols].copy()

        return env_df

    def add_fracdiff(self, d: float = 0.4, window: int = 50) -> "DataPreprocessor":
        weights = [1.0]
        for k in range(1, window):
            weights.append(-weights[-1] * (d - k + 1) / k)
        weights = np.array(weights)  # no need to reverse for np.convolve

        for ticker in self.tickers:
            col = f"Open_{ticker}"
            if col not in self.df.columns:
                continue
            series = self.df[col].values.astype(float)
            # 'valid' mode naturally produces NaNs for the first (window-1) rows
            convolved = np.convolve(series, weights, mode='full')[:len(series)].copy()
            convolved[:window - 1] = np.nan
            self.df[f"FracDiff_{ticker}"] = convolved

        return self

    def add_wavelet_denoise(self, wavelet: str = "db4", level: int = 1) -> "DataPreprocessor":
        for ticker in self.tickers:
            open_col = f"Open_{ticker}"
            if open_col not in self.df.columns:
                continue

            raw_returns = self.df[open_col].pct_change().fillna(0).values.copy()  # ← .copy() here

            coeffs = pywt.wavedec(raw_returns, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            coeffs[1:] = [pywt.threshold(c, sigma, mode="soft") for c in coeffs[1:]]
            denoised = pywt.waverec(coeffs, wavelet)[:len(raw_returns)]

            self.df[f"WaveletReturn_{ticker}"] = denoised
        return self

    def get_features_data(self) -> pd.DataFrame:
        """Returns only the engineered feature columns (FracDiff + WaveletReturn)."""
        feature_cols = [
            c for c in self.df.columns
            if c.startswith("FracDiff_") or c.startswith("WaveletReturn_")
        ]
        df_feat = self.df[feature_cols].copy()
        df_feat.dropna(inplace=True)
        return df_feat


class DataPipeline:
    """
    An all-in-one wrapper to provide a simple, clean interface for the RL Environment.
    It automatically handles downloading and preprocessing behind the scenes.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str, data_path: str = "data/raw"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_path = data_path

    def get_env_data(self, feature: str = 'Open') -> pd.DataFrame:
        """Downloads the data and filters it perfectly for the environment in one step."""
        # 1. Download internally
        downloader = DataDownloader(self.tickers, self.start_date, self.end_date, self.data_path)
        raw_df = downloader.fetch_data()

        # 2. Preprocess internally
        preprocessor = DataPreprocessor(raw_df, self.tickers)
        clean_df = preprocessor.get_env_ready_data(feature=feature)

        return clean_df

    def get_features_data(self, d: float = 0.4, fracdiff_window: int = 50,
                          use_fracdiff: bool = True, use_wavelet: bool = True) -> pd.DataFrame:
        downloader = DataDownloader(self.tickers, self.start_date, self.end_date, self.data_path)
        raw_df = downloader.fetch_data()
        preprocessor = DataPreprocessor(raw_df, self.tickers)
        if use_fracdiff:
            preprocessor.add_fracdiff(d=d, window=fracdiff_window)
        if use_wavelet:
            preprocessor.add_wavelet_denoise()
        return preprocessor.get_features_data()