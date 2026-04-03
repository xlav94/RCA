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

# Configuration parameters for the environment
stocks            = config.get('ENV','STOCKS').split(',')
window_size       = config.getint('ENV', 'WINDOW_SIZE')
env_name          = config.get('ENV', 'ENV_NAME')
objective         = config.get('ENV', 'OBJECTIVE')
normalize         = config.getboolean('ENV', 'NORMALIZE')

df = DataPipeline(tickers=stocks, start_date='2010-01-01', end_date='2026-02-28').get_env_data(feature='Open')
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = pd.concat([df_train.tail(window_size), df.iloc[train_size:]])

def make_env(df_env, stocks_env, objective_env, window_size_env, env_name_env, rank):
    def _init():
        custom_env =  CustomEnv(df_env, stocks_env, objective_env, window_size=window_size_env, env_name=env_name_env)
        custom_env.reset(seed=train_seed + rank)
        return custom_env
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
    env_fns = [make_env(df_train, stocks, objective, window_size, env_name, i) for i in range(num_cpu)]
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
            clip_reward=10.0,
        )

    if checkpoint:
        checkpoint_dir = f"models/{algo}_agent_checkpoints_seed_{train_seed}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}/"
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, 5_000_000 // num_cpu),
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
    env_test = CustomEnv(df_test, stocks, objective, window_size=window_size, env_name=f"{env_name}_test")
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

    writer.close()
    print(f"Test terminé. Profit final: {(total_cumulative_return - 1) * 100:.2f}%")
    print(f"Benchmark Buy-and-Hold: {(total_cum_return_hold - 1) * 100:.2f}%")


if __name__ == "__main__":
    algo_type = "PPO"   # SAC or PPO
    if is_training:
        seed_everything(train_seed)
        train(algo_type, train_seed=train_seed)
    else:
        env = 'models/mul_PMPT_20M_norm/envs/PPO_seed_4_env.pkl'
        model = 'models/mul_PMPT_30M/PPO_agent_checkpoints_seed_4_2026-04-02-17-56/PPO_agent_20000000_steps.zip'
        seed_everything(test_seed)
        test(algo_type, model, env, test_seed)

