import configparser
from datetime import datetime
import os
import random
import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch.utils.tensorboard import SummaryWriter

from src.model import get_agent
from src.data import DataPipeline
from src.env import CustomEnv

config = configparser.ConfigParser()
config.read('config.ini')

# Configuration parameters for the model
hidden_size_lstm = config.getint('MODEL', 'HIDDEN_SIZE_LSTM')
num_layers_lstm   = config.getint('MODEL', 'NUM_LAYERS_LSTM')
learning_rate     = config.getfloat('MODEL', 'LEARNING_RATE')
n_steps           = config.getint('MODEL', 'N_STEPS')
batch_size        = config.getint('MODEL', 'BATCH_SIZE')
n_epochs          = config.getint('MODEL', 'N_EPOCHS')
total_timesteps   = config.getint('MODEL', 'TOTAL_TIMESTEPS')
seed            = config.getint('MODEL', 'SEED')

# Configuration parameters for the environment
stocks      = config.get('ENV','STOCKS').split(',')
window_size = config.getint('ENV', 'WINDOW_SIZE')
env_name    = config.get('ENV', 'ENV_NAME')

df = DataPipeline(tickers=stocks, start_date='2010-01-01', end_date='2026-02-28').get_env_data(feature='Open')
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

def make_env(df, stocks, window_size, env_name):
    def _init():
        return CustomEnv(df, stocks, window_size=window_size, env_name=env_name)
    return _init

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train():
    num_cpu = 8

    vec_env = SubprocVecEnv([make_env(df_train, stocks, window_size, env_name) for _ in range(num_cpu)])

    vec_env = VecMonitor(vec_env)

    # Train the model
    model = get_agent(vec_env, hidden_size_lstm=hidden_size_lstm, num_layers_lstm=num_layers_lstm,
                      learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs)

    model.learn(progress_bar=True,
                    total_timesteps=total_timesteps
                    )
    model.save(f'models/ppo_agent_{datetime.now().strftime("%Y%m%d-%H%M")}')


def test(seed: int):
    env_test = CustomEnv(df_test, stocks, window_size=window_size, env_name=f"{env_name}_test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load('models/ppo_agent_PMPT_10M.zip', env=env_test, device=device)

    obs, _ = env_test.reset(seed=seed)
    done = False

    print(f"Début du test sur {len(df_test)} points de données...")
    df_benchmark = df_test.copy()
    for stock in stocks:
        df_benchmark[f'{stock}_ret'] = df_benchmark[f'Open_{stock}'].pct_change().fillna(0)
    total_cumulative_return = 1.0
    total_cum_return_hold = 1.0
    writer =    SummaryWriter(log_dir=f"./tensorboard_logs/test_results_{datetime.now().strftime('%Y%m%d-%H%M')}")
    step = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)

        total_cumulative_return *= (1 + float(reward))

        daily_market_return = df_benchmark[[f'{s}_ret' for s in stocks]].iloc[step].mean()
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
        done = terminated or truncated

    writer.close()
    print(f"Test terminé. Profit final: {(total_cumulative_return - 1) * 100:.2f}%")


if __name__ == "__main__":
    seed_everything(seed)
    test(seed)
