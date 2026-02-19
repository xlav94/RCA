import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List
import numpy as np

class DataDownloader:
    def __init__(self, tickers: List[str], start_date: str, end_date: str, data_path: str = "data/raw") -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_path = data_path

    def fetch_data(self)-> pd.DataFrame:
        filepath = os.path.join(self.data_path, 'stock_data.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            last_date = df.index[-1].date()

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

    def get_feature_matrix(self, features: List[str] = ['Open']) -> np.ndarray:
        filepath = os.path.join(self.data_path, 'stock_data.csv')
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        selected_columns = []
        for feature in features:
            for col in df.columns:
                if col.startswith(f"{feature}_"):
                    selected_columns.append(col)

        filtered_df = df[selected_columns]
        return filtered_df.values

# TO IGNORE !!!
if __name__ == "__main__":
    # 1. Setup parameters
    my_tickers = ["AAPL", "MSFT"]
    start = "2024-01-01"
    end = "2024-02-01"
    data_path = 'data/raw/'

    # 2. Initialize
    downloader = DataDownloader(tickers=my_tickers, start_date=start, end_date=end, data_path=data_path)

    # 3. Fetch and build the CSV
    print("Fetching data and building the master CSV...")
    df = downloader.fetch_data()
    print("\n--- Master DataFrame (First 5 Rows) ---")
    print(df.head())

    # 4. Test the Matrix Extraction!
    print("\n--- Testing Matrix Extraction: 'Open' only ---")
    open_matrix = downloader.get_feature_matrix(features=['Open'])
    print(open_matrix[:3])  # Print first 3 rows

    print("\n--- Testing Matrix Extraction: 'Open' AND 'Volume' ---")
    multi_matrix = downloader.get_feature_matrix(features=['Open', 'Volume'])
    print(multi_matrix[:3])  # Print first 3 rows