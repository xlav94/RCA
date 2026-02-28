import unittest

import pandas as pd

from src.env import CustomEnv
import numpy as np


class TestCustomEnv(unittest.TestCase):

    def setUp(self):
        data = {
            'Asset_1': [100, 110],
            'Asset_2': [100, 90]
        }
        self.df = pd.DataFrame(data)
        self.env = CustomEnv(self.df, np.array(['Asset_1', 'Asset_2']), window_size=1)

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

    def test_calculate_reward(self):
        weights_equal = np.array([0.5, 0.5])
        reward_a = self.env._calculate_reward(weights_equal)
        assert reward_a == 0.0, f"Erreur 50/50: attendu 0.0, reçu {reward_a}"

        weights_win = np.array([1.0, 0.0])
        reward_b = self.env._calculate_reward(weights_win)
        assert np.isclose(reward_b, 0.10), f"Erreur 100% Win: attendu 0.10, reçu {reward_b}"

        weights_loss = np.array([0.0, 1.0])
        reward_c = self.env._calculate_reward(weights_loss)
        assert np.isclose(reward_c, -0.10), f"Erreur 100% Loss: attendu -0.10, reçu {reward_c}"

    def test_step_logic(self):
        """Vérifie le déroulement d'une étape (incrémentation, reward, termination)."""
        # On initialise l'env (current_step = window_size, ex: 1)
        obs_init, _ = self.env.reset()
        initial_step = self.env.current_step

        # Action fictive (logits)
        action = np.array([0.5, 0.5], dtype=np.float32)

        # Exécution du step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Vérification de l'incrémentation
        self.assertEqual(self.env.current_step, initial_step + 1,
                         "Le current_step n'a pas été incrémenté.")

        # 2. Vérification de la structure de l'observation
        self.assertIn("market_history", obs)
        self.assertIn("portfolio_state", obs)

        # 3. Vérification du reward (calculé entre t=0 et t=1 avec poids 0.5/0.5)
        # Prix Asset_1: 100 -> 110 (+10%), Asset_2: 100 -> 90 (-10%)
        self.assertAlmostEqual(reward, 0.0, places=7)

    def test_termination(self):
        """Vérifie que l'épisode s'arrête bien à la fin du DataFrame."""
        # On force le step juste avant la fin
        # len(df) = 2, donc l'index max est 1.
        self.env.current_step = len(self.df) - 1

        action = np.array([0.5, 0.5], dtype=np.float32)
        _, _, terminated, _, _ = self.env.step(action)

        self.assertTrue(terminated, "L'environnement devrait être terminé à la fin du DF.")