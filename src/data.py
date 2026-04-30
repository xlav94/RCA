import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from src.feature_engineering import build_fracdiff_features

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

            if pd.to_datetime(self.start_date).date() < first_date:
                print("Older start date requested. Re-downloading full dataset...")
                df = yf.download(self.tickers, start=self.start_date, end=self.end_date)
                df.columns = ['_'.join(col) for col in df.columns]
                df.to_csv(filepath)
                return df

            if last_date < pd.to_datetime(self.end_date).date():
                new_data = yf.download(self.tickers, start=last_date + timedelta(days=1), end=self.end_date)
                if not new_data.empty:
                    new_data.columns = ['_'.join(col) for col in new_data.columns]
                    df = pd.concat([df, new_data])
                    df.to_csv(filepath)
            return df

        else:
            df = yf.download(self.tickers, start=self.start_date, end=self.end_date)
            df.columns = ['_'.join(col) for col in df.columns]
            df.to_csv(filepath)
            return df


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

    def _get_raw(self) -> pd.DataFrame:
        return DataDownloader(self.tickers, self.start_date, self.end_date, self.data_path).fetch_data()

    def get_env_data(self, feature: str = 'Open') -> pd.DataFrame:
        """Returns the raw price matrix ready for the environment."""
        raw_df = self._get_raw()

        cash_col_name = f"{feature}_CASH"
        initial_price = 10.0
        annual_rate = 0.05

        # Calcul du multiplicateur quotidien (Intérêts composés)
        daily_multiplier = (1 + annual_rate) ** (1 / 252)

        # Création de la courbe de prix parfaite
        steps = np.arange(len(raw_df))
        raw_df[cash_col_name] = initial_price * (daily_multiplier ** steps)

        ordered_cols = [f"{feature}_{ticker}" for ticker in self.tickers]
        ordered_cols.append(cash_col_name)
        for col in ordered_cols:
            if col not in raw_df.columns:
                raise ValueError(f"Missing required column: {col}")
        return raw_df[ordered_cols].copy()

    def get_features_data(self, d: float = None, fracdiff_window: int = 50) -> pd.DataFrame:
        """Returns the FracDiff feature matrix. d=None triggers per-ticker ADF search."""
        return build_fracdiff_features(self._get_raw(), self.tickers, d=d, window=fracdiff_window)