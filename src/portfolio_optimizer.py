from typing import Callable
import jax.numpy as jnp
import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
from jax import jit
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box_section

###### Not functional yet ##########
@jit
def objective_mpt_jax(weights, mu, cov_matrix, l=2.0):
    port_return = jnp.dot(weights, mu)
    port_risk = 0.5 * l * jnp.dot(weights.T, jnp.dot(cov_matrix, weights))
    return -(port_return - port_risk)
#####################################

@jit
def objective_pmpt_jax(weights, mu, returns, l=1.0, epsilon=1e-5):
    port_return = jnp.dot(weights, mu)
    historical_port_return = jnp.dot(returns, weights)
    downside_risk = jnp.sqrt(jnp.mean(jnp.minimum(0, historical_port_return) ** 2) + epsilon)
    return -(port_return - l * downside_risk)

@jit(static_argnames=['objective', 'lower_bound', 'upper_bound'])
def minimize_jax(objective, initial_weights, mu, returns, lower_bound, upper_bound):
    low = jnp.full_like(initial_weights, lower_bound)
    high = jnp.full_like(initial_weights, upper_bound)
    w_coeffs = jnp.ones_like(initial_weights)
    c_target = 1.0
    my_hyperparams = (low, high, w_coeffs, c_target)

    def projection_box_section_custom(x, _unused_hyperparams=None):
        return projection_box_section(x, my_hyperparams, check_feasible=False)

    pg = ProjectedGradient(fun=objective,
                           projection=projection_box_section_custom,
                           stepsize=0.1,
                           maxiter=100,
                           tol=1e-9,
                           acceleration=True)

    return pg.run(initial_weights,
            mu=mu,
            returns=returns).params

class PortfolioOptimizer:
    def __init__(self, lower_bound : float=0.0, upper_bound : float=0.1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def minimize(self, objective : Callable, initial_weights_np : np.ndarray, mu_np : np.ndarray, returns_df : pd.DataFrame) -> np.ndarray:
        returns_jax = jnp.array(returns_df.values)
        mu_jax = jnp.array(mu_np)
        initial_weights_jax = jnp.array(initial_weights_np)
        weights = minimize_jax(objective, initial_weights_jax, mu_jax, returns_jax,
                            self.lower_bound, self.upper_bound)
        return np.array(weights)