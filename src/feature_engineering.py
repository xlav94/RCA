import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from typing import List

def _build_weights(d: float, window: int) -> np.ndarray:
    """Builds the binomial weight vector for fractional differentiation."""
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    return np.array(weights)

def find_optimal_d(series: np.ndarray, train_end: int,
                   d_range: np.ndarray = np.arange(0.1, 1.0, 0.05),
                   window: int = 50) -> float:
    """
    Finds the minimum d such that the fracdiff series is stationary
    on the training portion only (ADF test at 95% confidence).
    Fitting on train only prevents data leakage into test.
    """
    train_series = np.log(series[:train_end] + 1e-8)

    for d in d_range:
        weights  = _build_weights(d, window)
        convolved = np.convolve(train_series, weights, mode='full')[:len(train_series)]
        convolved[:window - 1] = np.nan
        clean = convolved[~np.isnan(convolved)]

        if len(clean) < 20:
            continue

        pval = adfuller(clean, autolag='AIC')[1]
        if pval < 0.05:
            return round(float(d), 2)

    return 1.0

def apply_fracdiff(series: np.ndarray, d: float, window: int = 50) -> np.ndarray:
    """
    Applies fractional differentiation to a price series with a given d.
    Returns an array of the same length with NaN for the first (window-1) rows.
    """
    weights   = _build_weights(d, window)
    convolved = np.convolve(series, weights, mode='full')[:len(series)].copy()
    convolved[:window - 1] = np.nan
    return convolved

def build_fracdiff_features(df: pd.DataFrame, tickers: List[str],
                             d = None, window: int = 50,
                             train_ratio: float = 0.8) -> pd.DataFrame:
    """
    Builds a FracDiff feature DataFrame for all tickers.
    d can be:
      - None        → finds optimal d per ticker via ADF test
      - float       → uses the same d for all tickers (e.g. 0.4)
      - dict        → uses per-ticker d from config (e.g. {'AAPL': 0.65, ...})
    """
    train_end = int(len(df) * train_ratio)
    result    = pd.DataFrame(index=df.index)

    for ticker in tickers:
        col = f"Open_{ticker}"
        if col not in df.columns:
            continue

        series = df[col].values.astype(float)

        if isinstance(d, dict):
            optimal_d = d.get(ticker, None)
            if optimal_d is None:
                print(f"  {ticker}: no d in config, running ADF search...")
                optimal_d = find_optimal_d(series, train_end, window=window)
        elif d is None:
            optimal_d = find_optimal_d(series, train_end, window=window)
        else:
            optimal_d = d

        print(f"  {ticker}: d = {optimal_d}")
        result[f"FracDiff_{ticker}"] = apply_fracdiff(series, optimal_d, window)

    result.dropna(inplace=True)
    return result

def plot_fracdiff_diagnostic(df: pd.DataFrame, ticker: str,
                              window: int = 50,
                              d_range: np.ndarray = np.arange(0.0, 1.05, 0.05),
                              train_ratio: float = 0.8,
                              save_path: str = None):
    """
    For a given ticker, plots:
    1. ADF statistic vs d — where stationarity is achieved
    2. Pearson correlation vs d — how much memory is preserved
    3. Original log price vs fracdiff at optimal d
    """
    col          = f"Open_{ticker}"
    series       = np.log(df[col].values.astype(float) + 1e-8)
    train_end    = int(len(series) * train_ratio)
    train_series = series[:train_end]

    adf_stats    = []
    correlations = []
    critical_value = -2.86

    for d in d_range:
        weights   = _build_weights(d, window)
        convolved = np.convolve(train_series, weights, mode='full')[:len(train_series)]
        convolved[:window - 1] = np.nan
        clean = convolved[~np.isnan(convolved)]

        if len(clean) < 20:
            adf_stats.append(np.nan)
            correlations.append(np.nan)
            continue

        adf_stats.append(adfuller(clean, autolag='AIC')[0])
        correlations.append(np.corrcoef(train_series[window - 1:], clean)[0, 1])

    optimal_d = None
    for d, stat in zip(d_range, adf_stats):
        if not np.isnan(stat) and stat < critical_value:
            optimal_d = d
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"FracDiff Diagnostic — {ticker}", fontsize=13)

    ax1 = axes[0]
    ax1.plot(d_range, adf_stats, color='black', label='ADF Statistic')
    ax1.axhline(critical_value, linestyle='--', color='gray', label='95% Critical Value (-2.86)')
    if optimal_d is not None:
        ax1.axvline(optimal_d, linestyle=':', color='red', label=f'Optimal d = {optimal_d}')
    ax1.set_xlabel('d')
    ax1.set_ylabel('ADF Statistic')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(d_range, correlations, color='blue', alpha=0.5, label='Pearson Correlation')
    ax2.set_ylabel('Correlation with Original', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    if optimal_d is not None:
        full_convolved = apply_fracdiff(series, optimal_d, window)
        ax3       = axes[1]
        ax3_right = ax3.twinx()
        ax3.plot(series, color='black', label='Log price (original)', alpha=0.7)
        ax3_right.plot(full_convolved, color='gray', label=f'FracDiff (d={optimal_d})', alpha=0.7)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Log Price', color='black')
        ax3_right.set_ylabel('FracDiff Value', color='gray')
        ax3.set_title(f'Original vs FracDiff at d={optimal_d}')
        ax3.legend(loc='upper left')
        ax3_right.legend(loc='lower right')

        idx = list(d_range).index(optimal_d)
        print(f"{ticker} — optimal d: {optimal_d}, correlation: {correlations[idx]:.4f}")

    plt.tight_layout()
    path = save_path or f"src/plots/fracdiff_diagnostic_{ticker}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()