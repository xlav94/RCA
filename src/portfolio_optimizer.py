from typing import Callable
import jax.numpy as jnp
import numpy as np
import pandas as pd
import jax
from sklearn.covariance import LedoitWolf
# jax.config.update("jax_enable_x64", True)
from jax import jit
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box_section

###### Not functional yet ##########
@jit
def objective_mpt_jax(weights, mu, returns, cov_matrix, l=2.0):
    port_return = jnp.dot(weights, mu)
    port_risk = 0.5 * l * jnp.dot(weights.T, jnp.dot(cov_matrix, weights))
    return -(port_return - port_risk)
#####################################

@jit
def objective_pmpt_jax(weights, mu, returns, cov_matrix, l=1.0, epsilon=1e-5):
    port_return = jnp.dot(weights, mu)
    historical_port_return = jnp.dot(returns, weights)
    downside_risk = jnp.sqrt(jnp.mean(jnp.minimum(0, historical_port_return) ** 2) + epsilon)
    return -(port_return - l * downside_risk)

@jit(static_argnames=['objective'])
def minimize_jax(objective, initial_weights, mu, returns, cov_matrix,
                 lower_bound, upper_bound):
    w_coeffs = jnp.ones_like(initial_weights)
    c_target = 1.0
    my_hyperparams = (lower_bound, upper_bound, w_coeffs, c_target)

    def projection_box_section_custom(x, _unused_hyperparams=None):
        return projection_box_section(x, my_hyperparams, check_feasible=False)

    pg = ProjectedGradient(fun=objective,
                           projection=projection_box_section_custom,
                           stepsize=0.1,
                           maxiter=500,
                           tol=1e-9,
                           acceleration=True)

    return pg.run(initial_weights,
            mu=mu,
            returns=returns,
            cov_matrix=cov_matrix).params

class PortfolioOptimizer:
    def __init__(self, lower_bound : float=0.0, upper_bound : float=0.30):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def minimize(self, objective : Callable, initial_weights_np : np.ndarray, mu_np : np.ndarray, returns_df : pd.DataFrame) -> np.ndarray:
        returns_jax = jnp.array(returns_df.values)
        mu_jax = jnp.array(mu_np)
        initial_weights_jax = jnp.array(initial_weights_np)
        lw = LedoitWolf().fit(returns_df)
        cov_matrix_jax = jnp.array(lw.covariance_)
        num_assets = len(mu_jax)
        lower_bounds_arr = jnp.full(num_assets, self.lower_bound)
        upper_bounds_arr = jnp.full(num_assets, self.upper_bound)
        upper_bounds_arr = upper_bounds_arr.at[-1].set(1.0)
        weights = minimize_jax(objective, initial_weights_jax, mu_jax, returns_jax,
                                cov_matrix_jax,
                                lower_bounds_arr, upper_bounds_arr)
        return np.array(weights)