import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from src.env import CustomEnv

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self,
                observation_space: spaces.Dict = None,
                hidden_size_lstm=168,
                num_layers_lstm=2,
                batch_first=True):

        super().__init__(observation_space, features_dim=1)

        if observation_space is None:
            raise ValueError("Observation space cannot be None")
        if "market_history" not in observation_space.spaces:
            raise ValueError("The observation space is missing 'market_history' field")
        if "portfolio_state" not in observation_space.spaces:
            raise ValueError("The observation space is missing 'portfolio_state' field")
        #if "balance" not in observation_space.spaces:
            #raise ValueError("The observation space is missing 'balance' field")

        # LSTM to extract features from the market history
        market_history_shape = observation_space.spaces['market_history'].shape[1]
        self.lstm = torch.nn.LSTM(input_size=market_history_shape, hidden_size=hidden_size_lstm,
                                  num_layers=num_layers_lstm, batch_first=batch_first)

        # Simple linear layer to process the portfolio state
        portfolio_state_shape = observation_space.spaces['portfolio_state'].shape[0]
        self.mlp_portfolio = torch.nn.Linear(portfolio_state_shape, portfolio_state_shape)
        self.relu = torch.nn.ReLU()

        # The balance is a single scalar value, so we can directly use it without additional processing
        #balance_shape = observation_space.spaces['balance'].shape[0]

        # Update the features dimension to reflect the combined output of the LSTM and MLP
        self._features_dim = hidden_size_lstm + portfolio_state_shape

    def forward(self, observations) -> torch.Tensor:
        if "market_history" not in observations:
            raise ValueError("The observation space is missing 'market_history' field")
        if "portfolio_state" not in observations:
            raise ValueError("The observation space is missing 'portfolio_state' field")
        #if "balance" not in observations:
            #raise ValueError("The observation space is missing 'balance' field")

        out, (h_c, c_n) = self.lstm(observations['market_history'])
        out_portfolio_state = self.relu(self.mlp_portfolio(observations['portfolio_state'])) # shape(
        #balance = observations['balance'].view(-1, 1)
        combined_features = torch.cat((h_c[-1], out_portfolio_state), dim=1)
        return combined_features




def get_agent(env, hidden_size_lstm=168, num_layers_lstm=2, batch_first=True,
              learning_rate=0.001, n_steps=2048, batch_size=50, n_epochs=10, seed=1):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"L'agent s'entraînera sur : {device}")

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
                                       hidden_size_lstm=hidden_size_lstm,
                                       num_layers_lstm=num_layers_lstm,
                                       batch_first=batch_first)
    )

    model = PPO("MultiInputPolicy", env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                seed=seed,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                device=device
                )
    return model