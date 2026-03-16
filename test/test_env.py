import unittest

import pandas as pd
from scipy.optimize import check_grad

from src.env import CustomEnv
import numpy as np


class TestCustomEnv(unittest.TestCase):

    def setUp(self):
        data = np.random.uniform(100, 200, (100, 5))
        self.df = pd.DataFrame(data, columns=[f'Stock_{i}' for i in range(5)])
        self.window_size = 10
        self.env = CustomEnv(self.df, self.df.columns, "PMPT", window_size=self.window_size)

    def test_reset_shapes(self):
        """Verify reset returns the correct dictionary structure and shapes."""
        obs, info = self.env.reset()

        # Check dictionary keys
        self.assertIn("market_history", obs)
        self.assertIn("portfolio_state", obs)

        # Check shapes
        self.assertEqual(obs["market_history"].shape, (self.window_size, self.env.num_assets))
        self.assertEqual(obs["portfolio_state"].shape, (self.env.num_assets,))

    def test_observation_values(self):
        """Verify the market_history window matches the dataframe slice."""
        obs, _ = self.env.reset()

        # The first observation should be from row 0 up to row 10 (exclusive)
        expected_window = self.df.iloc[0:self.window_size].values
        np.testing.assert_array_almost_equal(obs["market_history"], expected_window, decimal=2)

        # Random observation
        random_step = np.random.randint(self.window_size, len(self.df))
        self.env.current_step = random_step

        obs_random = self.env._get_observation()
        expected_random = self.df.iloc[random_step - self.window_size: random_step].values

        np.testing.assert_array_almost_equal(
            obs_random["market_history"],
            expected_random,
            decimal=2
        )

    def test_data_types(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs["market_history"].dtype, np.float32)
        self.assertEqual(obs["portfolio_state"].dtype, np.float32)

    def test_initial_state(self):
        """Verify initial financial conditions."""
        obs, _ = self.env.reset()
        self.assertTrue(np.all(obs["portfolio_state"] == 0))

    def test_get_weights_from_action(self):
        """Vérifie que la somme est toujours exactement 1.0, peu importe l'action."""
        test_actions = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([10.0, -5.0, 2.0, 0.0, 1.0]),
            np.ones(5),
            np.zeros(5)
        ]

        for action in test_actions:
            weights = self.env._get_weights_from_action(action)
            # Vérifie la dimension
            self.assertEqual(len(weights), 5)
            # Vérifie la somme
            self.assertAlmostEqual(np.sum(weights), 1.0, places=7,
                                   msg=f"La somme des poids n'est pas 1.0 pour l'action {action}")

    def test_calculate_reward(self):
        data = {
            'Asset_1': [100, 110],
            'Asset_2': [100, 90]
        }
        df = pd.DataFrame(data)
        env = CustomEnv(df, np.array(['Asset_1', 'Asset_2']), "PMPT", window_size=1)

        weights_equal = np.array([0.5, 0.5])
        reward_a = env._calculate_reward(weights_equal)
        assert reward_a == 0.0, f"Erreur 50/50: attendu 0.0, reçu {reward_a}"

        weights_win = np.array([1.0, 0.0])
        reward_b = env._calculate_reward(weights_win)
        assert np.isclose(reward_b, 0.10), f"Erreur 100% Win: attendu 0.10, reçu {reward_b}"

        weights_loss = np.array([0.0, 1.0])
        reward_c = env._calculate_reward(weights_loss)
        assert np.isclose(reward_c, -0.10), f"Erreur 100% Loss: attendu -0.10, reçu {reward_c}"

    def test_step_logic(self):
        """Vérifie le déroulement d'une étape (incrémentation, reward, termination)."""
        obs_init, _ = self.env.reset()
        initial_step = self.env.current_step

        # Action à 5 dimensions
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Vérification de l'incrémentation
        self.assertEqual(self.env.current_step, initial_step + 1)

        # 2. Vérification de la structure
        self.assertEqual(obs["market_history"].shape, (self.window_size, 5))
        self.assertEqual(obs["portfolio_state"].shape, (5,))

        # 3. Vérification du reward
        # Puisque les données sont aléatoires, on vérifie que c'est un float valide
        self.assertIsInstance(float(reward), float)

    def test_termination(self):
        """Vérifie que l'épisode s'arrête bien à la fin du DataFrame."""
        self.env.current_step = len(self.df) - 1

        # Action à 5 dimensions
        action = np.zeros(5, dtype=np.float32)
        _, _, terminated, _, _ = self.env.step(action)

        self.assertTrue(terminated, "L'environnement devrait être terminé à l'index final.")

    def test_get_weights_mpt_robustness(self):
        """Teste la validité des poids et le fallback en cas d'erreur."""
        self.env.current_step = 20

        # 1. TEST DE VALIDITÉ : Action normale
        action_valide = np.array([0.5, -0.2, 0.1, 0.8, -0.1])
        weights = self.env._get_weights_from_action_mpt(action_valide)

        # Vérifications
        self.assertIsInstance(weights, np.ndarray, "Doit retourner un np.ndarray")
        self.assertEqual(weights.shape, (5,), "La taille du vecteur doit être égale au nombre d'actifs")
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4, msg="La somme des poids doit être égale à 1")
        self.assertTrue(np.all(weights >= 0),
                        "Par défaut, Markowitz ne doit pas retourner de poids négatifs (Long-only)")

        # 2. TEST DE RÉSILIENCE : Action avec NaN (simule un crash du réseau de neurones)
        action_corrompue = np.array([np.nan, 1.0, 0.5, 0.2, 0.1])
        weights_fallback = self.env._get_weights_from_action_mpt(action_corrompue)

        # Vérification du Equal Weight (1/5 = 0.2)
        expected_fallback = np.full(5, 0.2)
        np.testing.assert_array_almost_equal(weights_fallback, expected_fallback,
                                             err_msg="Le fallback doit être un Equal-Weight en cas d'erreur")

    def test_pmpt_gradient(self):
        # 1. Simulation de données (21 actifs, 60 jours)
        np.random.seed(42)
        num_assets = 21
        window_size = 60

        # On simule des rendements (returns) et des prédictions (mu)
        global returns, mu  # Simule le contexte de ta classe
        returns = np.random.normal(0.001, 0.02, (window_size, num_assets))
        mu = np.random.normal(0.01, 0.05, num_assets)

        # 2. Tes fonctions (version corrigée et vectorisée)
        def objective_PMPT(weights, l=1.0, epsilon=1e-5):
            port_return = weights @ mu
            historical_port_return = returns @ weights
            downside_risk = np.sqrt(np.mean(np.square(np.clip(historical_port_return, None, 0))) + epsilon)
            return -(port_return - l * downside_risk)

        def jacobian_PMPT(weights, l=1.0, epsilon=1e-5):
            historical_port_return = returns @ weights
            downside_risk = np.sqrt(np.mean(np.square(np.clip(historical_port_return, None, 0))) + epsilon)
            grad_risk = (1 / (len(returns) * downside_risk)) * returns.T @ np.clip(historical_port_return, None, 0)
            return -(mu - l * grad_risk)

        # 3. Création de poids de test (doivent sommer à 1)
        test_weights = np.random.dirichlet(np.ones(num_assets))

        # 4. Calcul de l'erreur entre l'analytique et le numérique
        # check_grad renvoie la norme L2 de la différence
        error = check_grad(objective_PMPT, jacobian_PMPT, test_weights, 1.0, 1e-5)

        print("--- Rapport de Test Unitaire : Borey-Alpha ---")
        print(f"Erreur calculée : {error:.2e}")

        # 5. Assertion (Seuil de tolérance standard : 1e-6)
        if error < 1e-6:
            print("SUCCÈS : Le Jacobien est mathématiquement correct.")
        else:
            print("ÉCHEC : Il y a une erreur dans la formule du gradient.")

        # Vérification des dimensions (Crucial pour SLSQP)
        grad_shape = jacobian_PMPT(test_weights).shape
        print(f"Dimensions du gradient : {grad_shape} (Attendu : ({num_assets},))")