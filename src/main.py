import configparser

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from src.model import get_agent
from src.data import DataPipeline
from src.env import CustomEnv

config = configparser.ConfigParser()
config.read('../config.ini')

# Configuration parameters for the model
hidden_size_lstm = config.getint('MODEL', 'HIDDEN_SIZE_LSTM')
num_layers_lstm   = config.getint('MODEL', 'NUM_LAYERS_LSTM')
learning_rate     = config.getfloat('MODEL', 'LEARNING_RATE')
n_steps           = config.getint('MODEL', 'N_STEPS')
batch_size        = config.getint('MODEL', 'BATCH_SIZE')
n_epochs          = config.getint('MODEL', 'N_EPOCHS')
total_timesteps   = config.getint('MODEL', 'TOTAL_TIMESTEPS')

# Configuration parameters for the environment
stocks      = config.get('ENV','STOCKS').split(',')
window_size = config.getint('ENV', 'WINDOW_SIZE')
env_name    = config.get('ENV', 'ENV_NAME')

def train():
    df = DataPipeline(tickers=stocks, start_date='2010-01-01', end_date='2024-01-01').get_env_data(feature='Open')
    env = CustomEnv(df, stocks, window_size=window_size, env_name=env_name)

    # Train the model
    model = get_agent(env, hidden_size_lstm=hidden_size_lstm, num_layers_lstm=num_layers_lstm,
                      learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs)

    model.learn(progress_bar=True,
                    total_timesteps=total_timesteps
                    )
    model.save('models/ppo_agent')


def test():
    pipeline_test = DataPipeline(
        tickers=stocks,
        start_date='2024-01-02',
        end_date='2025-12-31'
    )
    df_test = pipeline_test.get_env_data(feature='Open')

    env_test = CustomEnv(df_test, stocks, window_size=window_size, env_name=f"{env_name}_test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load('models/ppo_agent', env=env_test, device=device)

    obs, _ = env_test.reset()
    done = False

    print(f"Début du test sur {len(df_test)} points de données...")

    total_cumulative_return = 1.0
    daily_returns = []
    writer =    SummaryWriter(log_dir="./tensorboard_logs/test_results")
    step = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)

        total_cumulative_return *= (1 + float(reward))

        writer.add_scalar("Performance/Daily_Reward", float(reward), step)
        writer.add_scalar("Performance/Total_Return", total_cumulative_return - 1, step)
        weights_dict = {stocks[i]: float(info["portfolio_weights"][i]) for i in range(len(stocks))}
        writer.add_scalars("Allocation/Portfolio_Weights", weights_dict, step)

        step += 1
        done = terminated or truncated

    writer.close()
    print(f"Test terminé. Profit final: {(total_cumulative_return - 1) * 100:.2f}%")


if __name__ == "__main__":
    train()