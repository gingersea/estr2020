"""
model.py
--------
Defines the Bayesian model for the course project.

Model
-----
  Parameters : theta = (mu, rho) in R^2, where rho = log(sigma).
  Prior      : theta ~ N(0, I_2)
  Likelihood : x_n | theta ~ N(mu, exp(2*rho)), i.i.d. for n=1,...,N
  Posterior  : p(theta | D) ∝ p(theta) * p(D | theta)   [analytically intractable]

The posterior is intractable because the change-of-variables rho = log(sigma)
results in a non-standard density that does not belong to any known conjugate family.

All log-probability functions are used throughout for numerical stability.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

def log_prior(theta: np.ndarray) -> float:
    """
    Log prior: log p(theta) = log N(theta; 0, I).

    Parameters
    ----------
    theta : ndarray, shape (2,)
        theta = [mu, rho], where rho = log(sigma).

    Returns
    -------
    float
        Log-prior value.
    """
    # N(0, I): log p = -0.5 * ||theta||^2 - d/2 * log(2*pi)
    d = theta.shape[0]
    return -0.5 * np.dot(theta, theta) - 0.5 * d * np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def log_likelihood(theta: np.ndarray, data: np.ndarray) -> float:
    """
    Log likelihood: log p(D | theta) = sum_n log N(x_n; mu, sigma^2).

    Parameters
    ----------
    theta : ndarray, shape (2,)
        theta = [mu, rho].
    data  : ndarray, shape (N,)
        Observed data.

    Returns
    -------
    float
        Log-likelihood value.
    """
    mu, rho = theta[0], theta[1]
    sigma = np.exp(rho)          # sigma = exp(rho) > 0
    N = data.shape[0]

    # log N(x; mu, sigma^2) = -0.5*log(2*pi) - log(sigma) - (x-mu)^2 / (2*sigma^2)
    ll = (
        -0.5 * N * np.log(2.0 * np.pi)
        - N * np.log(sigma)
        - 0.5 * np.sum((data - mu) ** 2) / (sigma ** 2)
    )
    return ll


# ---------------------------------------------------------------------------
# Unnormalized log-posterior (target)
# ---------------------------------------------------------------------------

def log_unnorm_posterior(theta: np.ndarray, data: np.ndarray) -> float:
    """
    Log unnormalized posterior: log p~(theta) = log p(theta) + log p(D | theta).

    This can be evaluated pointwise and serves as the target for both MCMC and VI.

    Parameters
    ----------
    theta : ndarray, shape (2,)
    data  : ndarray, shape (N,)

    Returns
    -------
    float
    """
    return log_prior(theta) + log_likelihood(theta, data)


# ---------------------------------------------------------------------------
# Predictive density for a new observation x*
# ---------------------------------------------------------------------------

def log_pred_density(x_star: float, theta: np.ndarray) -> float:
    """
    Log predictive density: log p(x* | theta) = log N(x*; mu, sigma^2).

    Parameters
    ----------
    x_star : float
        New observation.
    theta  : ndarray, shape (2,)

    Returns
    -------
    float
    """
    mu, rho = theta[0], theta[1]
    sigma = np.exp(rho)
    return (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(sigma)
        - 0.5 * (x_star - mu) ** 2 / sigma ** 2
    )


# ---------------------------------------------------------------------------
# Monte Carlo posterior predictive approximation
# ---------------------------------------------------------------------------

def mc_predictive(x_star: float, samples: np.ndarray) -> float:
    """
    Approximate posterior predictive via Monte Carlo integration:
        p(x* | D) ≈ (1/S) * sum_s p(x* | theta^(s))

    Parameters
    ----------
    x_star  : float
    samples : ndarray, shape (S, 2)
        Posterior samples theta^(s).

    Returns
    -------
    float
        Estimated predictive density at x_star.
    """
    log_preds = np.array([log_pred_density(x_star, s) for s in samples])
    # Use log-sum-exp for numerical stability
    max_lp = np.max(log_preds)
    return np.exp(max_lp) * np.mean(np.exp(log_preds - max_lp))
