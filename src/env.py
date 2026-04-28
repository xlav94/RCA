import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.special import softmax

from src.portfolio_optimizer import PortfolioOptimizer, objective_pmpt_jax, objective_mpt_jax


class CustomEnv(gym.Env):
    def __init__(self, df, stocks, objective : str, window_size=50, initial_balance=10000, env_name='RCA', df_features=None):
        self.stocks = stocks
        self.current_step = window_size
        self.df = df
        self.df_features = df_features
        self.objective = objective
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.env_name = env_name
        self.num_assets = len(stocks)
        self.weights = np.full(self.num_assets, 1 / self.num_assets)
        self.po = PortfolioOptimizer(lower_bound=0., upper_bound=0.10)

        self.action_space = spaces.Box(low=-1.0,
                                     high=1.0,
                                     shape=(self.num_assets,),
                                    dtype=np.float32)

        num_extra = len(df_features.columns) if df_features is not None else 0

        # Observation: A window of technical indicators
        self.observation_space = spaces.Dict({
            "market_history": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window_size, len(df.columns) + num_extra),  # ← only change
                dtype=np.float32
            ),
            "portfolio_state": spaces.Box(
                low=0, high=1,
                shape=(self.num_assets,),
                dtype=np.float32
            ),
        })

    def _get_observation(self) -> dict:
        # Slice the dataframe for the LSTM
        history_window = self.df.iloc[self.current_step - self.window_size: self.current_step].values

        if self.df_features is not None:
            features_window = self.df_features.iloc[
                self.current_step - self.window_size: self.current_step
            ].values
            history_window = np.concatenate([history_window, features_window], axis=1)

        return {
            "market_history": history_window.astype(np.float32),
            "portfolio_state": self.weights.astype(np.float32),
        }

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.weights = np.zeros(self.num_assets, dtype=np.float32)  # Start with 0% in assets (all cash)

        return self._get_observation(), {}

    def step(self, action : np.ndarray):
        portfolio_weights = self._get_weights_from_action_mpt(action)
        portfolio_return, transaction_penality, downside_penalty, log_returns = self._calculate_reward(portfolio_weights)
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
            "transaction_penality": transaction_penality,
            "downside_penalty": downside_penalty,
            "log_returns": log_returns,
        }
        return observation, reward, terminated, truncated, info

    def _get_weights_from_action_mpt(self, action, precision=3, lower_bound=0., upper_bound=0.10):
        try:
            if np.any(np.isnan(action)):
                raise ValueError("Action contient des NaN")

            prices = self.df.iloc[self.current_step - self.window_size: self.current_step]
            returns = prices.pct_change().dropna()
            mu = action
            num_assets = self.num_assets
            initial_weights = np.full(num_assets, 1 / num_assets)
            weights = None

            if self.objective == "MPT":
                weights = self.po.minimize(objective_mpt_jax,
                                           initial_weights,
                                           mu,
                                           returns)
            elif self.objective == "PMPT":
                weights = self.po.minimize(objective_pmpt_jax,
                                      initial_weights,
                                      mu,
                                      returns)

            elif self.objective == "SOFTMAX":
                weights = softmax(action)

            """weights = np.round(weights, decimals=precision)
            diff = 1.0 - np.sum(weights)
            weights[weights.argmax()] += diff"""

        except Exception as e:
            print(f"Error in MPT optimization: {e}")
            weights = np.full(self.num_assets, 1 / self.num_assets)

        return weights

    def _calculate_reward(self, portfolio_weights, penality_factor=0.0003):
        # On calcule le rendement quotidien du portefeuille en utilisant les poids et les rendements des actifs
        current_prices = self.df.iloc[self.current_step].values
        previous_prices = self.df.iloc[self.current_step - 1].values
        assets_returns = (current_prices - previous_prices) / previous_prices
        assets_log_returns = np.log(current_prices / previous_prices)
        log_returns = np.dot(portfolio_weights, assets_log_returns)
        portfolio_return = np.dot(portfolio_weights, assets_returns)
        downside_penalty = 0.5 * min(0, portfolio_return)**2

        # Penalite si changement de poids important (pour encourager la stabilité du portefeuille)
        if self.current_step == self.window_size:
            transaction_penality = 0
        else:
            weight_change = np.sum(np.abs(portfolio_weights - self.weights)) #L1 norm
            transaction_penality = weight_change * penality_factor
        return portfolio_return, transaction_penality, downside_penalty, log_returns\

    def render(self):
        return

    def close(self):
        return
