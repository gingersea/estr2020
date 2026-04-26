"""
data.py
-------
Toy dataset generation for the Bayesian approximate inference project.

True data-generating process
-----------------------------
  x_n ~ N(mu_true, sigma_true^2),  n = 1, ..., N

We use:
  mu_true    = 2.0
  sigma_true = 1.5  (rho_true = log(1.5) ≈ 0.405)
  N          = 100

The dataset is fixed via a seed so that experiments are reproducible.
"""

import numpy as np


TRUE_MU    = 2.0
TRUE_SIGMA = 1.5
TRUE_RHO   = np.log(TRUE_SIGMA)   # ≈ 0.405
N_SAMPLES  = 100
RANDOM_SEED = 42


def generate_data(
    n: int = N_SAMPLES,
    mu: float = TRUE_MU,
    sigma: float = TRUE_SIGMA,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Generate toy data from the true data-generating process.

    Parameters
    ----------
    n     : int   — number of observations
    mu    : float — true mean
    sigma : float — true standard deviation
    seed  : int   — random seed for reproducibility

    Returns
    -------
    ndarray, shape (n,)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)


if __name__ == "__main__":
    data = generate_data()
    print(f"Generated {len(data)} observations.")
    print(f"True  mu={TRUE_MU:.3f}, sigma={TRUE_SIGMA:.3f}, rho={TRUE_RHO:.3f}")
    print(f"Sample mean={data.mean():.3f}, sample std={data.std():.3f}")
