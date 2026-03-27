import configparser
from datetime import datetime
from typing import Callable

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self,
                observation_space: spaces.Dict = None,
                hidden_size_lstm=168,
                num_layers_lstm=2,
                lstm_dropout=0,
                batch_first=True,
                use_cnn=False):

        super().__init__(observation_space, features_dim=1)
        self.use_cnn = use_cnn

        if observation_space is None:
            raise ValueError("Observation space cannot be None")
        if "market_history" not in observation_space.spaces:
            raise ValueError("The observation space is missing 'market_history' field")
        if "portfolio_state" not in observation_space.spaces:
            raise ValueError("The observation space is missing 'portfolio_state' field")

        if self.use_cnn:
            # CNN to extract features from the market history
            self.cnn = torch.nn.Sequential(
                # Layer 1
                torch.nn.Conv1d(in_channels=observation_space.spaces["market_history"].shape[1],
                                out_channels=32,
                                kernel_size=3,
                                padding=1
                                ),
                torch.nn.BatchNorm1d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),

                # Layer 2
                torch.nn.Conv1d(in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                padding=1
                                ),
                torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),

                # Layer 3
                torch.nn.Conv1d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                padding=1
                                ),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(lstm_dropout),
            )
            market_history_shape = 128
        else:
            market_history_shape = observation_space.spaces['market_history'].shape[1]

        # LSTM to extract features from the market history
        self.lstm = torch.nn.LSTM(input_size=market_history_shape, hidden_size=hidden_size_lstm,
                                  num_layers=num_layers_lstm, batch_first=batch_first,
                                  dropout=lstm_dropout)

        # Simple linear layer to process the portfolio state
        portfolio_state_shape = observation_space.spaces['portfolio_state'].shape[0]
        self.mlp_portfolio = torch.nn.Linear(portfolio_state_shape, portfolio_state_shape)
        self.relu = torch.nn.ReLU()

        # Update the features dimension to reflect the combined output of the LSTM and MLP
        self._features_dim = hidden_size_lstm + portfolio_state_shape

    def forward(self, observations) -> torch.Tensor:
        if "market_history" not in observations:
            raise ValueError("The observation space is missing 'market_history' field")
        if "portfolio_state" not in observations:
            raise ValueError("The observation space is missing 'portfolio_state' field")

        if self.use_cnn:
            out_cnn = self.cnn(observations['market_history'].permute(0, 2, 1))
            out, (h_c, c_n) = self.lstm(out_cnn.permute(0, 2, 1))
        else:
            out, (h_c, c_n) = self.lstm(observations['market_history'])

        out_portfolio_state = self.relu(self.mlp_portfolio(observations['portfolio_state']))  # shape(
        combined_features = torch.cat((h_c[-1], out_portfolio_state), dim=1)
        return combined_features


config = configparser.ConfigParser()
config.read('config.ini')

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

def get_agent_ppo(env, hidden_size_lstm=168, num_layers_lstm=2, dropout_lstm=0, use_cnn=False, batch_first=True, seed=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device : {device}")

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
                                       hidden_size_lstm=hidden_size_lstm,
                                       num_layers_lstm=num_layers_lstm,
                                       lstm_dropout=dropout_lstm,
                                       batch_first=batch_first,
                                       use_cnn=use_cnn)
    )

    model = PPO("MultiInputPolicy", env,
                learning_rate=linear_schedule(config.getfloat('PPO', 'LEARNING_RATE')),
                n_steps=config.getint('PPO', 'N_STEPS'),
                batch_size=config.getint('PPO', 'BATCH_SIZE'),
                n_epochs=config.getint('PPO', 'N_EPOCHS'),
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=f"./tensorboard_logs/PPO_{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
                device=device,
                seed=seed
                )
    return model

def get_agent_sac(env, hidden_size_lstm=168, num_layers_lstm=2, dropout_lstm=0, use_cnn=False, batch_first=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"L'agent s'entraînera sur : {device}")

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
                                       hidden_size_lstm=hidden_size_lstm,
                                       num_layers_lstm=num_layers_lstm,
                                       lstm_dropout=dropout_lstm,
                                       batch_first=batch_first,
                                       use_cnn=use_cnn)
    )

    model = SAC("MultiInputPolicy", env,
                learning_rate=linear_schedule(config.getfloat('SAC', 'LEARNING_RATE')),
                batch_size=config.getint('SAC', 'BATCH_SIZE'),
                buffer_size=config.getint('SAC', 'BUFFER_SIZE'),
                gradient_steps=config.getint('SAC', 'GRADIENT_STEPS'),
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=f"./tensorboard_logs/SAC_{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
                device=device
                )
    return model