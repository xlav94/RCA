import configparser
from datetime import datetime
import os
import random
import numpy as np
import pandas as pd

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from src.model import get_agent_ppo, get_agent_sac
from src.data import DataPipeline
from src.env import CustomEnv

config = configparser.ConfigParser()
config.read('config.ini')

# Configuration parameters for the model
is_training       = config.getboolean('MODEL', 'IS_TRAINING')
hidden_size_lstm  = config.getint('MODEL', 'HIDDEN_SIZE_LSTM')
num_layers_lstm   = config.getint('MODEL', 'NUM_LAYERS_LSTM')
dropout_lstm      = config.getfloat('MODEL', 'DROPOUT_LSTM')
use_cnn           = config.getboolean('MODEL', 'USE_CNN')
total_timesteps   = config.getint('MODEL', 'TOTAL_TIMESTEPS')
train_seed        = config.getint('MODEL', 'TRAIN_SEED')
test_seed         = config.getint('MODEL', 'TEST_SEED')
num_cpu           = config.getint('MODEL', 'NUM_CPU')
checkpoint        = config.getboolean('MODEL', 'CHECKPOINT')
use_fracdiff = config.getboolean('ENV', 'USE_FRACDIFF')

# Configuration parameters for the environment
stocks            = config.get('ENV','STOCKS').split(',')
window_size       = config.getint('ENV', 'WINDOW_SIZE')
env_name          = config.get('ENV', 'ENV_NAME')
objective         = config.get('ENV', 'OBJECTIVE')
normalize         = config.getboolean('ENV', 'NORMALIZE')

# Configuration parameters for fracdiff
fracdiff_d = {ticker: config.getfloat('FRACDIFF', ticker)
              for ticker in stocks}

pipeline = DataPipeline(tickers=stocks, start_date='2010-01-01', end_date='2026-02-28')
df       = pipeline.get_env_data(feature='Open')

if use_fracdiff:
    df_features = pipeline.get_features_data(d=fracdiff_d, fracdiff_window=50)
    common_idx  = df.index.intersection(df_features.index)
    df          = df.loc[common_idx]
    df_features = df_features.loc[common_idx]
else:
    df_features = None

train_size    = int(len(df) * 0.8)
df_train      = df.iloc[:train_size]
df_test       = pd.concat([df_train.tail(window_size), df.iloc[train_size:]])

if use_fracdiff:
    df_feat_train = df_features.iloc[:train_size]
    df_feat_test  = pd.concat([df_features.iloc[:train_size].tail(window_size),
                                df_features.iloc[train_size:]])
else:
    df_feat_train = None
    df_feat_test  = None


def make_env(df_env, df_feat_env, stocks_env, objective_env, window_size_env, env_name_env, rank):
    def _init():
        custom_env =  CustomEnv(df_env, stocks_env, objective_env, window_size=window_size_env, env_name=env_name_env, df_features=df_feat_env)
        custom_env.reset(seed=train_seed + rank)
        return custom_env
    return _init

def seed_everything(seed_init: int):
    random.seed(seed_init)
    # os.environ['PYTHONHASHSEED'] = str(seed_init)
    np.random.seed(seed_init)
    torch.manual_seed(seed_init)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_init)
        torch.cuda.manual_seed_all(seed_init)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

def train(algo, train_seed=None):
    env_fns = [make_env(df_train, df_feat_train, stocks, objective, window_size, env_name, i) for i in range(num_cpu)]
    if num_cpu > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    if normalize:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_reward=20,
        )

    if checkpoint:
        checkpoint_dir = f"models/{algo}_{objective}_agent_checkpoints_seed_{train_seed}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}/"
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, 10_000_000 // num_cpu),
            save_path=checkpoint_dir,
            name_prefix=f"{algo}_agent",
            save_replay_buffer=True,
        )
    else:
        checkpoint_callback = None

    # Train the model
    if algo == 'PPO':
        print("PPO selected for training.")
        model = get_agent_ppo(vec_env, hidden_size_lstm=hidden_size_lstm, num_layers_lstm=num_layers_lstm, seed=train_seed)

        model.learn(progress_bar=True,
                    total_timesteps=total_timesteps,
                    callback=checkpoint_callback
                        )
        model.save(f'models/ppo_agent_{total_timesteps}_seed_{train_seed}_{datetime.now().strftime("%Y-%m-%d-%H-%M")}')

    elif algo == 'SAC':
        print("SAC selected for training.")
        model = get_agent_sac(vec_env, hidden_size_lstm=hidden_size_lstm, num_layers_lstm=num_layers_lstm)

        model.learn(progress_bar=True,
                    total_timesteps=total_timesteps,
                    callback=checkpoint_callback
                    )
        model.save(f'models/sac_agent_{total_timesteps}_{datetime.now().strftime("%Y-%m-%d-%H-%M")}')

    if normalize:
        vec_env.save(f'models/envs/{algo}_seed_{train_seed}_env.pkl')


def test(algo, model_path, env_path, seed: int):
    env_test = CustomEnv(df_test, stocks, objective, window_size=window_size, env_name=f"{env_name}_test", df_features=df_feat_test)
    env_test.reset(seed=seed)

    if normalize:
        env_test = DummyVecEnv([lambda: env_test])
        env_test = VecNormalize.load(env_path, env_test)
        env_test.training = False
        env_test.norm_reward = False
        obs = env_test.reset()
    else:
        obs, _ = env_test.reset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if algo == 'PPO':
        model = PPO.load(model_path, env=env_test, device=device)
    elif algo == 'SAC':
        model = SAC.load(model_path, env=env_test, device=device)

    done = False

    print(f"Début du test sur {len(df_test)} points de données...")
    df_benchmark = df_test.copy()
    for stock in stocks:
        df_benchmark[f'{stock}_ret'] = df_benchmark[f'Open_{stock}'].pct_change().fillna(0)
    total_cumulative_return = 1.0
    total_cum_return_hold = 1.0
    writer =    SummaryWriter(log_dir=f"./tensorboard_logs/test_results_{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
    step = 0
    step_daily_return = window_size

    portfolio_returns_history = []
    benchmark_returns_history = []
    peak_agent = 1.0
    peak_bench = 1.0
    risk_free_rate = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        if normalize:
            obs, reward, dones, info = env_test.step(action)
            info = info[0]
            done = dones[0]
        else:
            obs, reward, terminated, truncated, info = env_test.step(action)
            done = terminated or truncated

        total_cumulative_return *= (1 + info["portfolio_return"])

        daily_market_return = df_benchmark[[f'{s}_ret' for s in stocks]].iloc[step_daily_return].mean()
        total_cum_return_hold *= (1 + daily_market_return)
        portfolio_returns_history.append(info["portfolio_return"])
        benchmark_returns_history.append(daily_market_return)
        # --- CALCUL DU MAXIMUM DRAWDOWN (En temps réel) ---
        if total_cumulative_return > peak_agent:
            peak_agent = total_cumulative_return
        if total_cum_return_hold > peak_bench:
            peak_bench = total_cum_return_hold

        dd_agent = (total_cumulative_return - peak_agent) / peak_agent
        dd_bench = (total_cum_return_hold - peak_bench) / peak_bench

        writer.add_scalars("Risk/Drawdown", {
            "Agent_PPO": dd_agent,
            "Buy_and_Hold": dd_bench
        }, step)
        # ----------------------------------------------------

        # --- CALCUL DU SHARPE ET SORTINO RATIO ---
        # On attend d'avoir au moins 10 jours de données pour éviter de diviser par zéro ou d'avoir des stats aberrantes
        if step > 10:
            # Métriques Agent
            agent_arr = np.array(portfolio_returns_history)
            mean_agent = np.mean(agent_arr) - risk_free_rate
            std_agent = np.std(agent_arr) + 1e-8
            downside_agent = agent_arr[agent_arr < 0]
            downside_std_agent = np.std(downside_agent) + 1e-8 if len(downside_agent) > 0 else 1e-8

            sharpe_agent = (mean_agent / std_agent) * np.sqrt(252)
            sortino_agent = (mean_agent / downside_std_agent) * np.sqrt(252)

            # Métriques Benchmark
            bench_arr = np.array(benchmark_returns_history)
            mean_bench = np.mean(bench_arr) - risk_free_rate
            std_bench = np.std(bench_arr) + 1e-8
            downside_bench = bench_arr[bench_arr < 0]
            downside_std_bench = np.std(downside_bench) + 1e-8 if len(downside_bench) > 0 else 1e-8

            sharpe_bench = (mean_bench / std_bench) * np.sqrt(252)
            sortino_bench = (mean_bench / downside_std_bench) * np.sqrt(252)

            writer.add_scalars("Risk_Adjusted_Returns/Sharpe_Ratio", {
                "Agent_PPO": sharpe_agent,
                "Buy_and_Hold": sharpe_bench
            }, step)

            writer.add_scalars("Risk_Adjusted_Returns/Sortino_Ratio", {
                "Agent_PPO": sortino_agent,
                "Buy_and_Hold": sortino_bench
            }, step)
        # ----------------------------------------------------
        writer.add_scalars("Comparison/Cumulative_Return", {
            "Agent_PPO": total_cumulative_return - 1,
            "Buy_and_Hold": total_cum_return_hold - 1
        }, step)

        writer.add_scalar("Performance/Daily_return", info["portfolio_return"], step)
        writer.add_scalar("Performance/Transaction_penality", info["transaction_penality"], step)
        weights_dict = {stocks[i]: float(info["portfolio_weights"][i]) for i in range(len(stocks))}
        weights_dict["Cash"] = float(info["portfolio_weights"][-1])
        writer.add_scalars("Allocation/Portfolio_Weights", weights_dict, step)

        step += 1
        step_daily_return += 1

    writer.close()
    print(f"Test terminé. Profit final: {(total_cumulative_return - 1) * 100:.2f}%")
    print(f"Benchmark Buy-and-Hold: {(total_cum_return_hold - 1) * 100:.2f}%")


if __name__ == "__main__":
    algo_type = "PPO"   # SAC or PPO
    if is_training:
        seed_everything(train_seed)
        train(algo_type, train_seed=train_seed)
    else:
        env = 'models/PPO_seed_42_env.pkl'
        model = 'models/ppo_agent_10000000_seed_42_2026-05-02-19-22.zip'
        seed_everything(test_seed)
        test(algo_type, model, env, test_seed)
