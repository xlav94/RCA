import os
import configparser
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import torch
from setuptools.sandbox import save_path
from stable_baselines3 import PPO

from src.data import DataPipeline
from src.env import CustomEnv


sns.set_theme(style="whitegrid")


def load_config(config_path: str = "/config.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)
    #print(config.sections())
    model_cfg = {
        "seed": config.getint("MODEL", "SEED"),
    }

    env_cfg = {
        "stocks": config.get("ENV", "STOCKS").split(","),
        "window_size": config.getint("ENV", "WINDOW_SIZE"),
        "env_name": config.get("ENV", "ENV_NAME"),
    }

    return model_cfg, env_cfg


def run_backtest(model_path: str, df_test: pd.DataFrame, stocks: list[str], window_size: int, env_name: str):
    """
    Rejoue le test complet et retourne un DataFrame contenant :
    - reward agent
    - cumulative return agent
    - daily return benchmark
    - cumulative return benchmark
    - portfolio weights
    """
    env = CustomEnv(df_test, stocks, window_size=window_size, env_name=f"{env_name}_visualization")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, env=env, device=device)

    obs, _ = env.reset()
    done = False

    records = []
    step = 0

    # Rendements benchmark par actif
    benchmark_df = df_test.copy()
    for stock in stocks:
        benchmark_df[f"{stock}_ret"] = benchmark_df[f"Open_{stock}"].pct_change().fillna(0.0)

    agent_cum = 1.0
    benchmark_cum = 1.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # reward
        agent_daily_return = float(reward)
        agent_cum *= (1.0 + agent_daily_return)

        # IMPORTANT:
        # Le step RL commence à current_step = window_size.
        # Donc pour comparer correctement au benchmark,
        # on aligne sur l'index window_size + step.
        benchmark_idx = window_size + step
        if benchmark_idx < len(benchmark_df):
            benchmark_daily_return = benchmark_df[[f"{s}_ret" for s in stocks]].iloc[benchmark_idx].mean()
        else:
            benchmark_daily_return = 0.0

        benchmark_cum *= (1.0 + benchmark_daily_return)

        row = {
            "step": step,
            "agent_daily_return": agent_daily_return,
            "agent_cumulative_return": agent_cum - 1.0,
            "benchmark_daily_return": float(benchmark_daily_return),
            "benchmark_cumulative_return": benchmark_cum - 1.0,
            "portfolio_return": float(info["portfolio_return"]),
            "transaction_penality": float(info["transaction_penality"]),
        }

        # Ajouter les poids de portefeuille
        for i, stock in enumerate(stocks):
            row[f"weight_{stock}"] = float(info["portfolio_weights"][i])

        records.append(row)

        step += 1
        done = terminated or truncated

    results_df = pd.DataFrame(records)
    return results_df


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


def plot_daily_profit(results_df: pd.DataFrame, save_path: str = None, show: bool = True):
    """
    Histogramme / densité simple des rendements journaliers de l'agent.
    """
    plt.figure(figsize=(10, 5))

    sns.histplot(results_df["agent_daily_return"], bins=40, kde=True)

    plt.title("Distribution of Agent Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


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


def main():
    model_cfg, env_cfg = load_config("../config.ini")

    stocks = env_cfg["stocks"]
    window_size = env_cfg["window_size"]
    env_name = env_cfg["env_name"]

    # Mets ici le chemin de ton modèle
    model_path = "../models/ppo_agent_PMPT_10M.zip"

    # Même logique que dans main.py
    df = DataPipeline(
        tickers=stocks,
        start_date="2010-01-01",
        end_date="2026-02-28"
    ).get_env_data(feature="Open")

    train_size = int(len(df) * 0.8)
    df_test = df.iloc[train_size:].copy()

    results_df = run_backtest(
        model_path=model_path,
        df_test=df_test,
        stocks=stocks,
        window_size=window_size,
        env_name=env_name
    )

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    results_csv_path = f"plots/backtest_results_{timestamp}.csv"
    results_df.to_csv(results_csv_path, index=False)

    plot_cumulative_returns(
        results_df,
        save_path=f"plots/cumulative_return_{timestamp}.png",
        show=True
    )

    plot_daily_profit(
        results_df,
        save_path=f"plots/daily_return_distribution_{timestamp}.png",
        show=True
    )

    plot_portfolio_weights_interactive(
        results_df,
        stocks=stocks,
        save_path_html=f"plots/portfolio_weights_interactive_{timestamp}.html"
    )

    final_agent_return = results_df["agent_cumulative_return"].iloc[-1] * 100
    final_benchmark_return = results_df["benchmark_cumulative_return"].iloc[-1] * 100

    print(f"Résultats sauvegardés dans : {results_csv_path}")
    print(f"Profit final agent PPO      : {final_agent_return:.2f}%")
    print(f"Profit final buy-and-hold   : {final_benchmark_return:.2f}%")


if __name__ == "__main__":
    main()