import configparser
from datetime import datetime
import os
import random
import numpy as np
import pandas as pd

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch.utils.tensorboard import SummaryWriter
from src.model import get_agent_ppo, get_agent_sac
from src.data import DataPipeline
from src.env import CustomEnv

config = configparser.ConfigParser()
config.read('config.ini')

# Configuration parameters for the model
hidden_size_lstm  = config.getint('MODEL', 'HIDDEN_SIZE_LSTM')
num_layers_lstm   = config.getint('MODEL', 'NUM_LAYERS_LSTM')
dropout_lstm      = config.getfloat('MODEL', 'DROPOUT_LSTM')
use_cnn           = config.getboolean('MODEL', 'USE_CNN')
total_timesteps   = config.getint('MODEL', 'TOTAL_TIMESTEPS')
train_seed        = config.getint('MODEL', 'TRAIN_SEED')
test_seed         = config.getint('MODEL', 'TEST_SEED')
num_cpu           = config.getint('MODEL', 'NUM_CPU')
checkpoint        = config.getboolean('MODEL', 'CHECKPOINT')

# Configuration parameters for the environment
stocks      = config.get('ENV','STOCKS').split(',')
window_size = config.getint('ENV', 'WINDOW_SIZE')
env_name    = config.get('ENV', 'ENV_NAME')
objective   = config.get('ENV', 'OBJECTIVE')

df = DataPipeline(tickers=stocks, start_date='2010-01-01', end_date='2026-02-28').get_env_data(feature='Open')
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = pd.concat([df_train.tail(window_size), df.iloc[train_size:]])

def make_env(df_env, stocks_env, objective_env, window_size_env, env_name_env):
    def _init():
        return CustomEnv(df_env, stocks_env, objective_env, window_size=window_size_env, env_name=env_name_env)
    return _init

def seed_everything(seed_init: int):
    random.seed(seed_init)
    os.environ['PYTHONHASHSEED'] = str(seed_init)
    np.random.seed(seed_init)
    torch.manual_seed(seed_init)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_init)
        torch.cuda.manual_seed_all(seed_init)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(algo, train_seed=None):

    vec_env = SubprocVecEnv([make_env(df_train, stocks, objective, window_size, env_name) for _ in range(num_cpu)])

    vec_env = VecMonitor(vec_env)

    if checkpoint:
        checkpoint_dir = f"models/{algo}_agent_checkpoints_seed_{train_seed}_{datetime.now().strftime('%Y-%m-%d-%H:%M')}/"
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
        model.save(f'models/ppo_agent_{total_timesteps}_{datetime.now().strftime("%Y-%m-%d-%H:%M")}')

    elif algo == 'SAC':
        print("SAC selected for training.")
        model = get_agent_sac(vec_env, hidden_size_lstm=hidden_size_lstm, num_layers_lstm=num_layers_lstm)

        model.learn(progress_bar=True,
                    total_timesteps=total_timesteps,
                    callback=checkpoint_callback
                    )
        model.save(f'models/sac_agent_{total_timesteps}_{datetime.now().strftime("%Y-%m-%d-%H:%M")}')


def test(seed: int):
    env_test = CustomEnv(df_test, stocks, objective, window_size=window_size, env_name=f"{env_name}_test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load('models/ppo_agent_10000000_2026-03-22-0343.zip', env=env_test, device=device)
    # model = SAC.load('models/SAC_agent_20000000_steps.zip', env=env_test, device=device)

    obs, _ = env_test.reset(seed=seed)
    done = False

    print(f"Début du test sur {len(df_test)} points de données...")
    df_benchmark = df_test.copy()
    for stock in stocks:
        df_benchmark[f'{stock}_ret'] = df_benchmark[f'Open_{stock}'].pct_change().fillna(0)
    total_cumulative_return = 1.0
    total_cum_return_hold = 1.0
    writer =    SummaryWriter(log_dir=f"./tensorboard_logs/test_results_{datetime.now().strftime('%Y-%m-%d-%H:%M')}")
    step = 0
    step_daily_return = window_size

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)

        total_cumulative_return *= (1 + float(reward))

        daily_market_return = df_benchmark[[f'{s}_ret' for s in stocks]].iloc[step_daily_return].mean()
        total_cum_return_hold *= (1 + daily_market_return)
        writer.add_scalars("Comparison/Cumulative_Return", {
            "Agent_PPO": total_cumulative_return - 1,
            "Buy_and_Hold": total_cum_return_hold - 1
        }, step)

        writer.add_scalar("Performance/Daily_return", info["portfolio_return"], step)
        writer.add_scalar("Performance/Transaction_penality", info["transaction_penality"], step)
        weights_dict = {stocks[i]: float(info["portfolio_weights"][i]) for i in range(len(stocks))}
        writer.add_scalars("Allocation/Portfolio_Weights", weights_dict, step)

        step += 1
        step_daily_return += 1
        done = terminated or truncated

    writer.close()
    print(f"Test terminé. Profit final: {(total_cumulative_return - 1) * 100:.2f}%")


if __name__ == "__main__":
    seed_everything(test_seed)
    test(test_seed)
    #train("PPO", train_seed=train_seed) # SAC or PPO