import math
import os
import configparser
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


from src.main import df_test

sns.set_theme(style="whitegrid")


def load_config(config_path: str = "/config.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)
    #print(config.sections())
    model_cfg = {
        "seed": config.getint("MODEL", "TEST_SEED"),
    }

    env_cfg = {
        "stocks": config.get("ENV", "STOCKS").split(","),
        "window_size": config.getint("ENV", "WINDOW_SIZE"),
        "env_name": config.get("ENV", "ENV_NAME"),
    }

    return model_cfg, env_cfg

def extract_weights_from_subdirs(parent_dir):
    all_assets_data = {}

    # On parcourt chaque sous-dossier dans le répertoire parent
    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)

        # On ne traite que les dossiers qui concernent l'allocation
        if os.path.isdir(folder_path) and "Allocation_Portfolio_Weights" in folder:
            asset_name = folder.split('_')[-1]  # Récupère le nom (ex: AMZN)

            # Charger l'accumulateur pour ce dossier spécifique
            event_acc = EventAccumulator(folder_path)
            event_acc.Reload()

            # Dans ces sous-dossiers, le tag est souvent simplifié
            # On cherche le tag de scalaire disponible
            tags = event_acc.Tags()['scalars']
            if tags:
                tag = tags[0]  # Généralement il n'y en a qu'un par dossier
                scalars = event_acc.Scalars(tag)
                steps = [e.step for e in scalars]
                values = [e.value for e in scalars]

                all_assets_data[f"weight_{asset_name}"] = pd.Series(values, index=steps)

    df = pd.DataFrame(all_assets_data)
    df.index.name = 'step'
    return df.reset_index()

def extract_root_scalar(log_dir: str, tag: str, column_name: str) -> pd.DataFrame:
    """
    Extrait un scalaire stocké directement dans le dossier principal TensorBoard.
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    available_tags = event_acc.Tags().get("scalars", [])
    if tag not in available_tags:
        raise ValueError(f"Tag '{tag}' introuvable dans {log_dir}. Tags disponibles: {available_tags}")

    scalars = event_acc.Scalars(tag)
    return pd.DataFrame({
        "step": [e.step for e in scalars],
        column_name: [e.value for e in scalars]
    })


def extract_scalar_from_subdir(parent_dir: str, folder_name: str, column_name: str) -> pd.DataFrame:
    """
    Extrait un scalaire depuis un sous-dossier TensorBoard spécifique.
    Ex: Comparison_Cumulative_Return_Agent_PPO
    """
    folder_path = os.path.join(parent_dir, folder_name)
    if not os.path.isdir(folder_path):
        raise ValueError(f"Sous-dossier introuvable: {folder_path}")

    event_acc = EventAccumulator(folder_path)
    event_acc.Reload()

    tags = event_acc.Tags().get("scalars", [])
    if not tags:
        raise ValueError(f"Aucun tag scalaire trouvé dans {folder_path}")

    tag = tags[0]
    scalars = event_acc.Scalars(tag)
    return pd.DataFrame({
        "step": [e.step for e in scalars],
        column_name: [e.value for e in scalars]
    })

def build_results_df_from_tensorboard(log_dir: str) -> pd.DataFrame:
    """
    Reconstruit results_df depuis TensorBoard uniquement.
    """
    # Racine
    df_agent_daily = extract_root_scalar(
        log_dir,
        tag="Performance/Daily_return",
        column_name="agent_daily_return"
    )

    df_penalty = extract_root_scalar(
        log_dir,
        tag="Performance/Transaction_penality",
        column_name="transaction_penality"
    )

    # Sous-dossiers Comparison/*
    df_agent_cum = extract_scalar_from_subdir(
        log_dir,
        folder_name="Comparison_Cumulative_Return_Agent_PPO",
        column_name="agent_cumulative_return"
    )

    df_benchmark_cum = extract_scalar_from_subdir(
        log_dir,
        folder_name="Comparison_Cumulative_Return_Buy_and_Hold",
        column_name="benchmark_cumulative_return"
    )

    # Sous-dossiers Allocation/*
    df_weights = extract_weights_from_subdirs(log_dir)

    # Merge
    results_df = df_agent_daily.merge(df_penalty, on="step", how="outer")
    results_df = results_df.merge(df_agent_cum, on="step", how="outer")
    results_df = results_df.merge(df_benchmark_cum, on="step", how="outer")
    results_df = results_df.merge(df_weights, on="step", how="outer")

    results_df = results_df.sort_values("step").reset_index(drop=True)

    # Reconstruire benchmark_daily_return à partir du cumulative benchmark
    wealth_benchmark = 1.0 + results_df["benchmark_cumulative_return"]
    results_df["benchmark_daily_return"] = wealth_benchmark / wealth_benchmark.shift(1) - 1.0
    results_df.loc[0, "benchmark_daily_return"] = np.nan

    return results_df

def extract_cumulative_return(log_dir, agent=True):
    if agent:
        folder = "Comparison_Cumulative_Return_Agent_PPO"
        col = "agent_cumulative_return"
    else:
        folder = "Comparison_Cumulative_Return_Buy_and_Hold"
        col = "benchmark_cumulative_return"

    path = os.path.join(log_dir, folder)

    event_acc = EventAccumulator(path)
    event_acc.Reload()

    tag = event_acc.Tags()["scalars"][0]
    scalars = event_acc.Scalars(tag)

    return pd.DataFrame({
        "step": [e.step for e in scalars],
        col: [e.value for e in scalars]
    })

def build_mean_from_tensorboard(log_dirs):
    dfs = []

    for i, log_dir in enumerate(log_dirs):
        df_agent = extract_cumulative_return(log_dir, agent=True)

        df_agent = df_agent.rename(columns={
            "agent_cumulative_return": f"agent_model_{i}"
        })

        dfs.append(df_agent)

    # Merge sur step
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="step", how="inner")

    agent_cols = [c for c in merged.columns if c.startswith("agent_model_")]

    # Mean + std
    merged["agent_cumulative_return_mean"] = merged[agent_cols].mean(axis=1)
    merged["agent_cumulative_return_std"] = merged[agent_cols].std(axis=1)

    # Benchmark (prendre depuis le premier run)
    df_benchmark = extract_cumulative_return(log_dirs[0], agent=False)
    merged = pd.merge(merged, df_benchmark, on="step", how="inner")

    return merged

def plot_normalized_vs_raw(mean_raw_df, mean_norm_df, save_path=None):
    plt.figure(figsize=(12, 6))

    # agent raw cum_return
    sns.lineplot(
        x=mean_raw_df["step"],
        y=mean_raw_df["agent_cumulative_return_mean"],
        label="PPO mean (raw env)"
    )

    # agent norm cum_return
    sns.lineplot(
        x=mean_norm_df["step"],
        y=mean_norm_df["agent_cumulative_return_mean"],
        label="PPO mean (normalized env)"
    )

    # Benchmark cum_return
    sns.lineplot(
        x=mean_raw_df["step"],
        y=mean_raw_df["benchmark_cumulative_return"],
        label="Buy & Hold",
        linestyle="--"
    )

    plt.axhline(0, linestyle="--", color="black")
    plt.title("Mean Cumulative Return: Raw vs Normalized Environments")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_cumulative_returns(results_df: pd.DataFrame, save_path: str = None, show: bool = True):
    """
    Trace la performance cumulée de l'agent vs benchmark.
    """
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=results_df,
        x="step",
        y="agent_cumulative_return",
        label="Agent PPO"
    )

    sns.lineplot(
        data=results_df,
        x="step",
        y="benchmark_cumulative_return",
        label="Buy & Hold equally weighted"
    )

    plt.title("Cumulative Return: PPO Agent vs Benchmark")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Return")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def build_mean_cumulative_return_df(results_dfs):
    """
    Build a DataFrame with:
    - step
    - mean cumulative return across models
    - std cumulative return across models
    - benchmark cumulative return
    """
    merged = results_dfs[0][["step", "benchmark_cumulative_return"]].copy()

    for i, df in enumerate(results_dfs):
        merged = merged.merge(
            df[["step", "agent_cumulative_return"]].rename(
                columns={"agent_cumulative_return": f"agent_cumulative_return_model_{i+1}"}
            ),
            on="step",
            how="inner"
        )

    agent_cols = [c for c in merged.columns if c.startswith("agent_cumulative_return_model_")]

    merged["agent_cumulative_return_mean"] = merged[agent_cols].mean(axis=1)
    merged["agent_cumulative_return_std"] = merged[agent_cols].std(axis=1)

    return merged

def plot_portfolio_weights_interactive(results_df, stocks, save_path_html=None):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    # Traces normales, chacune avec sa propre couleur
    for i, stock in enumerate(stocks):
        fig.add_trace(
            go.Scatter(
                x=results_df["step"],
                y=results_df[f"weight_{stock}"],
                mode="lines",
                name=stock,
                line=dict(
                    color=colors[i % len(colors)],
                    width=2.5
                ),
                opacity=1.0,
                line_shape="hv",   # plus lisible pour des poids qui changent par paliers
                hovertemplate=(
                    f"<b>{stock}</b><br>"
                    "Step: %{x}<br>"
                    "Weight: %{y:.4f}<extra></extra>"
                )
            )
        )



    fig.update_layout(
        title="Portfolio Allocation Over Time",
        xaxis_title="Time Step",
        yaxis_title="Weight",
        template="plotly_white",
        hovermode="x unified",
        width=1500,
        height=750,
        legend_title="Assets",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ],
        margin=dict(l=60, r=220, t=80, b=60)
    )

    # Poids bornés à 0.10 dans ton env
    fig.update_yaxes(range=[0, 0.105], tickformat=".3f")
    fig.update_xaxes(rangeslider_visible=True)

    if save_path_html is not None:
        fig.write_html(save_path_html)

    fig.show()

def plot_cumulative_transaction_cost(results_df, save_path=None, show=True):
    """
    Plot cumulative transaction penalty over time.
    """
    plot_df = results_df.copy()
    plot_df["cumulative_transaction_cost"] = plot_df["transaction_penality"].cumsum()

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=plot_df,
        x="step",
        y="cumulative_transaction_cost",
        linewidth=2
    )

    plt.title("Cumulative Transaction Cost")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Transaction Cost")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def plot_risk_return_scatter(results_df, df_test, stocks, window_size, save_path=None, show=True):
    """
    Risk-return scatter plot comparing:
    - PPO agent
    - equally weighted buy-and-hold benchmark
    - individual assets

    x-axis: annualized volatility
    y-axis: annualized return
    """
    ann_factor = 252

    # Agent returns
    agent_returns = results_df["agent_daily_return"].values

    # Benchmark returns aligned with RL timeline
    benchmark_df = df_test.copy()
    for stock in stocks:
        benchmark_df[f"{stock}_ret"] = benchmark_df[f"Open_{stock}"].pct_change().fillna(0.0)

    benchmark_returns = []
    for step in range(len(results_df)):
        idx = window_size + step
        if idx < len(benchmark_df):
            r = benchmark_df[[f"{s}_ret" for s in stocks]].iloc[idx].mean()
            benchmark_returns.append(float(r))

    benchmark_returns = np.array(benchmark_returns)

    points = []

    # PPO
    agent_ann_return = (1 + np.mean(agent_returns))**ann_factor - 1
    agent_ann_vol = np.std(agent_returns, ddof=1) * np.sqrt(ann_factor)
    points.append({
        "Strategy": "PPO Agent",
        "Type": "Agent",
        "Annual Return": agent_ann_return,
        "Annual Volatility": agent_ann_vol
    })

    # Benchmark
    benchmark_ann_return = (1 + np.mean(benchmark_returns))**ann_factor - 1
    benchmark_ann_vol = np.std(benchmark_returns, ddof=1) * np.sqrt(ann_factor)
    points.append({
        "Strategy": "Equal Weight B&H",
        "Type": "Benchmark",
        "Annual Return": benchmark_ann_return,
        "Annual Volatility": benchmark_ann_vol
    })

    # Individual assets
    for stock in stocks:
        asset_returns = benchmark_df[f"{stock}_ret"].iloc[window_size:window_size + len(results_df)].values

        asset_ann_return = (1 + np.mean(asset_returns))**ann_factor - 1
        asset_ann_vol = np.std(asset_returns, ddof=1) * np.sqrt(ann_factor)

        points.append({
            "Strategy": stock,
            "Type": "Asset",
            "Annual Return": asset_ann_return,
            "Annual Volatility": asset_ann_vol
        })

    scatter_df = pd.DataFrame(points)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=scatter_df,
        x="Annual Volatility",
        y="Annual Return",
        hue="Type",
        style="Type",
        s=140
    )

    # Annotate points
    for _, row in scatter_df.iterrows():
        plt.text(
            row["Annual Volatility"] + 0.002,
            row["Annual Return"] + 0.002,
            row["Strategy"],
            fontsize=9
        )

    plt.title("Risk-Return Scatter")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return scatter_df

def compute_rolling_sharpe_sortino(results_df, window=60, risk_free_rate=0.0, ann_factor=252):
    """
    Rolling Sharpe and Sortino for both strategies.
    """
    rf_daily = risk_free_rate / ann_factor
    df = results_df.copy()

    def rolling_sharpe(series):
        def f(x):
            excess = x - rf_daily
            vol = np.std(excess, ddof=1)
            if vol == 0:
                return np.nan
            return np.sqrt(ann_factor) * np.mean(excess) / vol
        return series.rolling(window).apply(f, raw=True)

    def rolling_sortino(series):
        def f(x):
            excess = x - rf_daily
            downside = excess[excess < 0]
            if len(downside) < 2:
                return np.nan
            downside_std = np.std(downside, ddof=1)
            if downside_std == 0:
                return np.nan
            return np.sqrt(ann_factor) * np.mean(excess) / downside_std
        return series.rolling(window).apply(f, raw=True)

    df["agent_rolling_sharpe"] = rolling_sharpe(df["agent_daily_return"])
    df["benchmark_rolling_sharpe"] = rolling_sharpe(df["benchmark_daily_return"])

    df["agent_rolling_sortino"] = rolling_sortino(df["agent_daily_return"])
    df["benchmark_rolling_sortino"] = rolling_sortino(df["benchmark_daily_return"])

    return df

def plot_rolling_ratios(results_df, metric="sharpe", save_path=None, show=True):
    """
    Plot rolling Sharpe or Sortino ratio over time.
    metric: "sharpe" or "sortino"
    """
    if metric == "sharpe":
        agent_col = "agent_rolling_sharpe"
        benchmark_col = "benchmark_rolling_sharpe"
        title = "Rolling Sharpe Ratio"
    elif metric == "sortino":
        agent_col = "agent_rolling_sortino"
        benchmark_col = "benchmark_rolling_sortino"
        title = "Rolling Sortino Ratio"
    else:
        raise ValueError("metric must be 'sharpe' or 'sortino'")

    plt.figure(figsize=(12, 5))
    sns.lineplot(x=results_df["step"], y=results_df[agent_col], label="PPO Agent")
    sns.lineplot(x=results_df["step"], y=results_df[benchmark_col], label="Buy & Hold")

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Ratio")
    plt.axhline(0, linestyle="--", linewidth=1, color="black")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def compute_alpha(results_df, risk_free_rate=0.0, ann_factor=252):
    """
    Compute annualized alpha of agent vs benchmark.
    a = E[agent_return - Rf] - b * E[benchmark_return - Rf]
    b = cov(agent_return - Rf, benchmark_return - Rf) / var(benchmark_return - Rf)
    excess = return - Rf
    """

    agent_returns = results_df["agent_daily_return"].dropna().values
    benchmark_returns = results_df["benchmark_daily_return"].dropna().values

    # align lengths
    min_len = min(len(agent_returns), len(benchmark_returns))
    agent_returns = agent_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    rf_daily = risk_free_rate / ann_factor

    # excess returns
    agent_excess = agent_returns - rf_daily
    bench_excess = benchmark_returns - rf_daily

    # beta
    cov = np.cov(agent_excess, bench_excess)[0, 1]
    var = np.var(bench_excess)

    beta = cov / var if var != 0 else np.nan

    # alpha (daily)
    alpha_daily = np.mean(agent_excess) - beta * np.mean(bench_excess)

    # annualized alpha
    alpha_annual = alpha_daily * ann_factor

    return alpha_annual

def main():

    model_cfg, env_cfg = load_config("config.ini")

    stocks = env_cfg["stocks"]
    window_size = env_cfg["window_size"]

    log_path = "tensorboard_logs/test_results_seed_4_2026-03-27-16-55"
    results_df = build_results_df_from_tensorboard(log_path)


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plots_dir = f"src/plots/{timestamp}"
    os.makedirs(plots_dir, exist_ok=True)

    results_csv_path = f"{plots_dir}/tensorboard_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    results_df.head()

    plot_cumulative_returns(
        results_df,
        save_path=f"{plots_dir}/cumulative_return.png",
        show=True
    )

    log_dirs_norm = [
        "tensorboard_logs/test_results_seed_4_2026-03-27-15-51",
        "tensorboard_logs/test_results_seed_13_2026-03-27-16-04",
        "tensorboard_logs/test_results_seed_21_2026-03-27-16-00",
        "tensorboard_logs/test_results_seed_42_2026-03-27-17-06",
        "tensorboard_logs/test_results_seed_2300_2026-03-27-16-03",
    ]

    log_dirs_raw = [
        "tensorboard_logs/test_results_seed_4_2026-03-27-16-55",
        "tensorboard_logs/test_results_seed_13_2026-03-27-16-59",
        "tensorboard_logs/test_results_seed_21_2026-03-27-16-58",
        "tensorboard_logs/test_results_seed_42_2026-03-27-16-49",
        "tensorboard_logs/test_results_seed_2300_2026-03-27-16-58"
    ]

    mean_raw_df = build_mean_from_tensorboard(log_dirs_raw)
    mean_norm_df = build_mean_from_tensorboard(log_dirs_norm)

    mean_norm_df.to_csv(f"{plots_dir}/5_models_mean.csv", index=False)
    mean_raw_df.to_csv(f"{plots_dir}/5_models_mean_raw.csv", index=False)

    plot_normalized_vs_raw(
        mean_raw_df,
        mean_norm_df,
        save_path=f"{plots_dir}/raw_vs_normalized_mean_cumulative_return.png"
    )

    plot_portfolio_weights_interactive(
        results_df,
        stocks=stocks,
        save_path_html=f"{plots_dir}/portfolio_weights_interactive.html"
    )

    plot_cumulative_transaction_cost(
        results_df,
        save_path=f"{plots_dir}/cumulative_transaction_cost.png"
    )

    # Risk-return scatter
    scatter_df = plot_risk_return_scatter(
        results_df,
        df_test=df_test,
        stocks=stocks,
        window_size=window_size,
        save_path=f"{plots_dir}/risk_return_scatter.png"
    )
    scatter_df.to_csv(f"{plots_dir}/risk_return_scatter.csv", index=False)

    # Sortino and sharpe ratios
    rolling_df = compute_rolling_sharpe_sortino(results_df, window=60)

    plot_rolling_ratios(
        rolling_df,
        metric="sharpe",
        save_path=f"{plots_dir}/rolling_sharpe.png"
    )

    plot_rolling_ratios(
        rolling_df,
        metric="sortino",
        save_path=f"{plots_dir}/rolling_sortino.png"
    )

    alpha = compute_alpha(results_df)

    print(f"Alpha (annual): {alpha:.4f}")
    if alpha < 0:
        print("Agent sous-performe le buy&hold")
    elif alpha == 0:
        print("Agent performance ~ buy&hold performance")
    else :
        print("Agent sur-performe le sell&hold")

    final_agent_return = results_df["agent_cumulative_return"].iloc[-1] * 100
    final_benchmark_return = results_df["benchmark_cumulative_return"].iloc[-1] * 100

    print(f"Résultats sauvegardés dans : {results_csv_path}")
    print(f"Profit final agent PPO      : {final_agent_return:.2f}%")
    print(f"Profit final buy-and-hold   : {final_benchmark_return:.2f}%")


if __name__ == "__main__":
    main()