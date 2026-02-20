import unittest

import pandas as pd

from src.env import CustomEnv
import numpy as np


class TestCustomEnv(unittest.TestCase):

    def setUp(self):
        self.env = CustomEnv(pd.DataFrame())

    def test_get_weights_from_action(self):
        """Vérifie que la somme est toujours exactement 1.0, peu importe l'action."""
        test_actions = [
            np.array([0.1, 0.2, 0.3]),
            np.array([10.0, -5.0, 2.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.00001, 0.00002, 0.00003])
        ]

        for action in test_actions:
            weights = self.env._get_weights_from_action(action)
            self.assertAlmostEqual(np.sum(weights), 1.0, places=7,
                                   msg=f"La somme des poids n'est pas 1.0 pour l'action {action}")