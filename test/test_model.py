import unittest

from src.model import CustomCombinedExtractor
import numpy as np
import torch
from gymnasium import spaces

class TestCustomCombinedExtractor(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.hidden_size = 64
        self.n_assets = 3

        self.observation_space = spaces.Dict({
            'market_history': spaces.Box(low=0, high=1, shape=(10, 5), dtype=np.float32),
            'portfolio_state': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            #'balance': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

    def test_forward_pass(self):
        extractor = CustomCombinedExtractor(observation_space=self.observation_space,
                                            hidden_size_lstm=self.hidden_size)

        observations = {
            'market_history': torch.rand(self.batch_size, 10, 5),
            'portfolio_state': torch.rand(self.batch_size, 3),
            #'balance': torch.rand(self.batch_size, 1)
        }

        features = extractor(observations)

        expected_features_dim  = self.hidden_size + self.n_assets

        # Check dimensions of the output features
        self.assertEqual(features.shape[1], expected_features_dim)
        # Check that the batch size is preserved
        self.assertEqual(features.shape[0], self.batch_size)