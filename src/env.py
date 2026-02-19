import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RCAEnv(gym.Env):
    def __init__(self, df, window_size=50, initial_balance=10000, env_name='RCA'):
        self.weights = None
        self.balance = None
        self.current_step = None
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.env_name = env_name
        self.nb_actifs = 10

        #Actions -> 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.MultiDiscrete([3] * self.nb_actifs)

        #TODO: option2:
        # [-1, 1] for each asset:
        #       -1 = sell all
        #       -1 < x < -0.1 = sell
        #       -0.1 < x < 0.1 = hold
        #        0.1 < x < 1 = buy
        #        1 = buy with all available balance
        # self.action_space = spaces.Box(low=-1.0,
        #                               high=1.0,
        #                               shape=(self.nb_actifs,),
        #                              dtype=np.float32)

        # Observation: A window of technical indicators
        self.observation_space = spaces.Dict({
            "market_history": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window_size, len(df.columns)),
                dtype=np.float32
            ),
            "portfolio_state": spaces.Box(
                low=0, high=1,
                shape=(window_size,self.nb_actifs),  # Weights for Assets
                dtype=np.float32
            ),
            "balance": spaces.Box(low=0, high=np.inf, shape=(window_size,), dtype=np.float32)
        })

    def _get_observation(self):
        # Slice the dataframe for the LSTM
        history_window = self.df.iloc[self.current_step - self.window_size: self.current_step].values

        return {
            "market_history": history_window.astype(np.float32),
            "portfolio_state": self.weights.astype(np.float32),
            "balance": np.array([self.balance], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.weights = np.zeros(self.nb_actifs, dtype=np.float32)  # Start with 0% in assets (all cash)
        self.max_net_worth = self.initial_balance

        return self._get_observation(), {}

    def step(self, action):
        return

    def _calculate_reward(self, portfolio_return, weight_change, net_worth):
        return

    def render(self):
        return

    def close(self):
        return

