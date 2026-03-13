import unittest

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.env import CustomEnv
from src.model import CustomCombinedExtractor, get_agent
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
        })
        self.df = pd.DataFrame(np.random.rand(100, 5), columns=[f'Stock_{i}' for i in range(5)])
        self.env = CustomEnv(self.df, self.df.columns, window_size=10)

    def test_forward_pass(self):
        extractor = CustomCombinedExtractor(observation_space=self.observation_space,
                                            hidden_size_lstm=self.hidden_size,)

        observations = {
            'market_history': torch.rand(self.batch_size, 10, 5),
            'portfolio_state': torch.rand(self.batch_size, 3),
        }

        features = extractor(observations)

        expected_features_dim  = self.hidden_size + self.n_assets

        # Check dimensions of the output features
        self.assertEqual(features.shape[1], expected_features_dim)
        # Check that the batch size is preserved
        self.assertEqual(features.shape[0], self.batch_size)

    def test_env(self):
        try:
            check_env(self.env)
        except Exception as e:
            self.fail(f"Environment check failed: {e}")

    def test_ppo_forward_pass(self):
        """Vérifie que PPO peut lire l'observation et produire une action sans erreur."""
        # Initialisation du modèle avec ton architecture
        model = PPO("MultiInputPolicy", self.env, verbose=0)

        obs, _ = self.env.reset()

        # Test de la prédiction (Forward Pass)
        # Cela valide que ton LSTM et ton FeaturesExtractor fonctionnent
        try:
            action, _states = model.predict(obs, deterministic=True)
        except Exception as e:
            self.fail(f"Le modèle PPO a échoué lors de la prédiction: {e}")

        self.assertEqual(action.shape, (5,), "L'action prédite n'a pas la bonne dimension.")

    def test_training_start(self):
        """Vérifie que le modèle peut effectuer quelques étapes d'apprentissage (Backprop)."""
        model = PPO("MultiInputPolicy", self.env, n_steps=128, batch_size=64, verbose=0)

        # On teste juste 200 steps pour voir si les gradients ne sont pas 'NaN'
        try:
            model.learn(total_timesteps=200)
            success = True
        except Exception as e:
            success = False
            print(f"Erreur lors de l'apprentissage: {e}")

        self.assertTrue(success, "Le modèle n'a pas pu effectuer son cycle d'apprentissage.")

    def test_model_cnn(self):
        # 1. Utiliser les vraies dimensions de ton environnement
        window_size = 60
        n_assets = 21

        # On simule un observation_space réaliste
        obs_space = spaces.Dict({
            "market_history": spaces.Box(low=-1, high=1, shape=(window_size, n_assets), dtype=np.float32),
            "portfolio_state": spaces.Box(low=0, high=1, shape=(n_assets,), dtype=np.float32)
        })

        # 2. Initialisation de l'extracteur
        extractor = CustomCombinedExtractor(
            observation_space=obs_space,
            hidden_size_lstm=self.hidden_size,
            use_cnn=True
        )

        # 3. Création de données "mock" avec les bonnes dimensions (Batch, Time, Assets)
        observations = {
            'market_history': torch.rand(self.batch_size, window_size, n_assets),
            'portfolio_state': torch.rand(self.batch_size, n_assets),
        }

        # 4. Exécution
        features = extractor(observations)

        # 5. Calcul de la dimension attendue
        # hidden_size_lstm (sortie LSTM) + n_assets (sortie MLP portfolio)
        expected_features_dim = self.hidden_size + n_assets

        # --- ASSERTIONS ---
        # Vérifie que la dimension de sortie est correcte
        self.assertEqual(features.shape[1], expected_features_dim,
                         f"La dimension des features ({features.shape[1]}) ne correspond pas à l'attendu ({expected_features_dim})")

        # Vérifie que le batch size est préservé
        self.assertEqual(features.shape[0], self.batch_size)

        # Vérifie qu'il n'y a pas de valeurs NaN (important avec les CNN/LSTM)
        self.assertFalse(torch.isnan(features).any(), "L'extracteur a produit des valeurs NaN")

    def test_training_start_with_cnn(self):
        """Vérifie que le modèle peut effectuer quelques étapes d'apprentissage (Backprop)."""
        model = get_agent(self.env, hidden_size_lstm=self.hidden_size, use_cnn=True, n_steps=128, batch_size=64)

        try:
            model.learn(total_timesteps=200)
            success = True
        except Exception as e:
            success = False
            print(f"Erreur lors de l'apprentissage: {e}")

        self.assertTrue(success, "Le modèle n'a pas pu effectuer son cycle d'apprentissage.")