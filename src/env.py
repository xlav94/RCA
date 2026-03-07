import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pypfopt import EfficientFrontier, risk_models, objective_functions
from scipy.special import softmax

class CustomEnv(gym.Env):
    def __init__(self, df, stocks, window_size=50, initial_balance=10000, env_name='RCA'):
        self.stocks = stocks
        self.current_step = window_size
        self.df = df
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.env_name = env_name
        self.num_assets = len(stocks)
        self.weights = np.full(self.num_assets, 1 / self.num_assets)

        self.action_space = spaces.Box(low=-1.0,
                                     high=1.0,
                                     shape=(self.num_assets,),
                                    dtype=np.float32)

        # Observation: A window of technical indicators
        self.observation_space = spaces.Dict({
            "market_history": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window_size, len(df.columns)),
                dtype=np.float32
            ),
            "portfolio_state": spaces.Box(
                low=0, high=1,
                shape=(self.num_assets,),  # Weights for Assets
                dtype=np.float32
            ),
            #"balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

    def _get_observation(self) -> dict:
        # Slice the dataframe for the LSTM
        history_window = self.df.iloc[self.current_step - self.window_size: self.current_step].values

        return {
            "market_history": history_window.astype(np.float32),
            "portfolio_state": self.weights.astype(np.float32),
            #"balance": np.array([self.balance], dtype=np.float32)
        }

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.weights = np.zeros(self.num_assets, dtype=np.float32)  # Start with 0% in assets (all cash)

        return self._get_observation(), {}

    def step(self, action : np.ndarray):
        portfolio_weights = self._get_weights_from_action_mpt(action)
        portfolio_return, transaction_penality = self._calculate_reward(portfolio_weights)
        reward = portfolio_return - transaction_penality
        self.current_step += 1
        self.weights = portfolio_weights

        observation = self._get_observation()
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        info = {
            "portfolio_weights": portfolio_weights,
            "reward": reward,
            "portfolio_return": portfolio_return,
            "transaction_penality": transaction_penality
        }
        return observation, reward, terminated, truncated, info

    def _get_weights_from_action(self, action, precision=2):
        weights =  softmax(action)
        rounded_weights = np.round(weights, decimals=precision)
        diff = 1 - np.sum(rounded_weights)
        rounded_weights[rounded_weights.argmax()] += diff
        return rounded_weights

    def _get_weights_from_action_mpt(self, action, precision=3):
        try:
            prices = self.df.iloc[self.current_step - self.window_size: self.current_step]
            cov_matrix = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            ef = EfficientFrontier(action, cov_matrix)
            ef.add_objective(objective_functions.L2_reg, gamma=1)
            raw_weights = ef.max_sharpe()
            raw_weights_list = np.array(list(raw_weights.values()))
            weights = np.round(raw_weights_list, decimals=precision)
            diff = 1 - np.sum(weights)
            weights[weights.argmax()] += diff
        except Exception as e:
            print(f"Error in MPT optimization: {e}")
            weights = np.full(self.num_assets, 1 / self.num_assets)
        return weights

    def _calculate_reward(self, portfolio_weights, penality_factor=0.0003):
        # On calcule le rendement quotidien du portefeuille en utilisant les poids et les rendements des actifs
        current_prices = self.df.iloc[self.current_step].values
        previous_prices = self.df.iloc[self.current_step - 1].values
        asset_returns = (current_prices - previous_prices) / previous_prices
        portfolio_return = np.dot(portfolio_weights, asset_returns)

        # Penalite si changement de poids important (pour encourager la stabilité du portefeuille)
        if self.current_step == self.window_size:
            transaction_penality = 0
        else:
            weight_change = np.sum(np.abs(portfolio_weights - self.weights)) #L1 norm
            transaction_penality = weight_change * penality_factor
        return portfolio_return, transaction_penality

    def render(self):
        return

    def close(self):
        return

