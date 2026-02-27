import unittest
import pandas as pd
import numpy as np

from src.env import RCAEnv


class TestRCAEnv(unittest.TestCase):
    def setUp(self):
        # Create dummy financial data: 100 days, 5 stocks
        data = np.random.uniform(100, 200, (100, 5))
        self.df = pd.DataFrame(data, columns=[f'Stock_{i}' for i in range(5)])
        print("DataFrame Sample:\n", self.df.head())
        self.window_size = 10
        self.env = RCAEnv(self.df, window_size=self.window_size)

    def test_reset_shapes(self):
        """Verify reset returns the correct dictionary structure and shapes."""
        obs, info = self.env.reset()

        # Check dictionary keys
        self.assertIn("market_history", obs)
        self.assertIn("portfolio_state", obs)
        self.assertIn("balance", obs)

        # Check shapes
        self.assertEqual(obs["market_history"].shape, (self.window_size, self.env.nb_stocks))
        self.assertEqual(obs["portfolio_state"].shape, (self.env.nb_stocks,))
        self.assertEqual(obs["balance"].shape, (1,))

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
        self.assertEqual(obs["balance"].dtype, np.float32)

    def test_initial_state(self):
        """Verify initial financial conditions."""
        obs, _ = self.env.reset()
        self.assertEqual(self.env.balance, self.env.initial_balance)
        self.assertTrue(np.all(obs["portfolio_state"] == 0))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
