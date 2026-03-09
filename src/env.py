import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
import pandas as pd
from sklearn.covariance import LedoitWolf


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

    def _get_weights_from_action_mpt(self, action, precision=3, lower_bound=0.05, upper_bound=0.50):
        try:
            if np.any(np.isnan(action)):
                raise ValueError("Action contient des NaN")

            prices = self.df.iloc[self.current_step - self.window_size: self.current_step]
            returns = prices.pct_change().dropna()

            lw = LedoitWolf().fit(returns)
            cov_matrix = lw.covariance_
            mu = action

            num_assets = self.num_assets
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            bounds = tuple((lower_bound, upper_bound) for _ in range(num_assets))

            def objective_MPT(weights, l=2.0):
                port_return = np.dot(weights, mu)
                port_risk = 0.5 * l * np.dot(weights.T, np.dot(cov_matrix, weights))
                return -(port_return - port_risk)

            def objective_PMPT(weights, l=1.0):
                port_return = np.dot(weights, mu)
                historical_port_return = np.dot(returns, weights)
                downside_risk = np.mean(np.square(np.minimum(0, historical_port_return)))
                return -(port_return - l * downside_risk)

            initial_weights = np.full(num_assets, 1 / num_assets)

            result = minimize(objective_PMPT, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'ftol': 1e-7, 'maxiter': 100})

            # Si SLSQP échoue, on ne crash pas, on prend les poids actuels du résultat
            # ou on bascule sur le fallback.
            if not result.success:
                # Souvent, même si le "linesearch" échoue, result.x est une solution décente
                weights = result.x
            else:
                weights = result.x

            # 6. Post-traitement
            weights = np.round(weights, decimals=precision)
            diff = 1.0 - np.sum(weights)
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

