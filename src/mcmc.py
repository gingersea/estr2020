"""
mcmc.py
-------
Metropolis-Hastings MCMC for the Bayesian approximate inference project.

Algorithm
---------
Random-walk Metropolis-Hastings:
  1. Propose  theta' ~ N(theta^(t), sigma^2 * I)
  2. Compute  log A = min(0, log p~(theta') - log p~(theta^(t)))
  3. Accept   with probability exp(log A); otherwise stay at theta^(t).

The proposal is symmetric, so the Hastings correction cancels and we obtain
the standard Metropolis acceptance ratio.

After discarding B burn-in samples (and optionally thinning by factor `thin`),
we retain S posterior samples for downstream prediction and analysis.
"""

import numpy as np
from typing import Callable, Tuple, List


def metropolis_hastings(
    log_target: Callable[[np.ndarray], float],
    theta_init: np.ndarray,
    n_iter: int = 20_000,
    burn_in: int = 5_000,
    step_size: float = 0.1,
    thin: int = 1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run the Metropolis-Hastings algorithm.

    Parameters
    ----------
    log_target : callable
        Function that returns log p~(theta) (log unnormalized posterior).
    theta_init : ndarray, shape (d,)
        Initial parameter vector.
    n_iter     : int
        Total number of MCMC iterations (including burn-in).
    burn_in    : int
        Number of initial samples to discard.
    step_size  : float
        Standard deviation sigma of the isotropic Gaussian proposal.
    thin       : int
        Thinning factor (retain every `thin`-th post-burn-in sample).
    seed       : int
        Random seed for reproducibility.

    Returns
    -------
    samples      : ndarray, shape (S, d)
        Post-burn-in (and thinned) posterior samples.
    chain        : ndarray, shape (n_iter, d)
        Full chain including burn-in (for trace plots / diagnostics).
    accept_rate  : float
        Fraction of proposals accepted over the full run.
    """
    rng = np.random.default_rng(seed)
    d = theta_init.shape[0]

    chain = np.empty((n_iter, d))
    chain[0] = theta_init.copy()

    log_p_current = log_target(theta_init)
    n_accept = 0

    for t in range(1, n_iter):
        # --- Proposal step ---
        epsilon = rng.standard_normal(d)
        theta_proposal = chain[t - 1] + step_size * epsilon

        # --- Log acceptance ratio (log-space for numerical stability) ---
        log_p_proposal = log_target(theta_proposal)
        log_accept_ratio = log_p_proposal - log_p_current

        # --- Accept / reject ---
        log_u = np.log(rng.uniform())
        if log_u <= log_accept_ratio:
            chain[t] = theta_proposal
            log_p_current = log_p_proposal
            n_accept += 1
        else:
            chain[t] = chain[t - 1]

    accept_rate = n_accept / (n_iter - 1)

    # --- Discard burn-in and thin ---
    post_burn = chain[burn_in:]
    samples = post_burn[::thin]

    return samples, chain, accept_rate


def run_multiple_chains(
    log_target: Callable[[np.ndarray], float],
    d: int,
    n_chains: int = 3,
    n_iter: int = 20_000,
    burn_in: int = 5_000,
    step_size: float = 0.1,
    thin: int = 1,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Run multiple independent MH chains from overdispersed initializations.

    Each chain is initialized by drawing theta_init ~ N(0, 4*I) to ensure
    starting points are overdispersed relative to the posterior.

    Parameters
    ----------
    log_target : callable
    d          : int — parameter dimension
    n_chains   : int — number of independent chains
    n_iter, burn_in, step_size, thin : as in metropolis_hastings()
    seed       : int — base random seed (chain k uses seed + k)

    Returns
    -------
    all_samples     : list of ndarray, each shape (S, d)
    all_chains      : list of ndarray, each shape (n_iter, d)
    all_accept_rates: list of float
    """
    rng = np.random.default_rng(seed)
    all_samples, all_chains, all_accept_rates = [], [], []

    for k in range(n_chains):
        theta_init = rng.normal(0.0, 2.0, size=d)
        samples, chain, ar = metropolis_hastings(
            log_target=log_target,
            theta_init=theta_init,
            n_iter=n_iter,
            burn_in=burn_in,
            step_size=step_size,
            thin=thin,
            seed=seed + k + 1,
        )
        all_samples.append(samples)
        all_chains.append(chain)
        all_accept_rates.append(ar)

    return all_samples, all_chains, all_accept_rates
