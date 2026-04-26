"""
diagnostics.py
--------------
Convergence diagnostics for MCMC.

Implements:
  - Gelman-Rubin R-hat statistic (multivariate extension, per-parameter)
  - Helper functions for reporting mixing quality
"""

import numpy as np
from typing import List


def gelman_rubin(chains: List[np.ndarray]) -> np.ndarray:
    """
    Compute the Gelman-Rubin potential scale reduction factor (R-hat) for
    each parameter dimension.

    Given K chains each of length M (post-burn-in), the statistic is:

        W  = (1/K) * sum_k s_k^2         (within-chain variance)
        B  = M / (K-1) * sum_k (theta_k_bar - theta_bar)^2  (between-chain)
        V~ = (1 - 1/M) * W + (1/M) * B  (pooled variance estimate)
        R  = sqrt(V~ / W)

    Convergence is indicated when R < 1.1 for all parameters.

    Parameters
    ----------
    chains : list of ndarray, each shape (M, d)
        Post-burn-in chain samples from K independent chains.

    Returns
    -------
    r_hat : ndarray, shape (d,)
        Per-parameter R-hat values.
    """
    K = len(chains)
    M = chains[0].shape[0]
    d = chains[0].shape[1]

    # Per-chain means and variances
    chain_means = np.array([c.mean(axis=0) for c in chains])   # (K, d)
    chain_vars  = np.array([c.var(axis=0, ddof=1) for c in chains])  # (K, d)

    # Grand mean (over all chains)
    grand_mean = chain_means.mean(axis=0)   # (d,)

    # Within-chain variance W
    W = chain_vars.mean(axis=0)             # (d,)

    # Between-chain variance B
    B = M / (K - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)  # (d,)

    # Pooled marginal posterior variance estimate
    V_tilde = (1.0 - 1.0 / M) * W + (1.0 / M) * B  # (d,)

    # R-hat
    r_hat = np.sqrt(V_tilde / np.where(W > 0, W, 1e-12))  # avoid /0

    return r_hat


def effective_sample_size(chain: np.ndarray) -> np.ndarray:
    """
    Estimate the effective sample size (ESS) for each parameter using
    the autocorrelation sum estimator.

    ESS = M / (1 + 2 * sum_{k=1}^{K_max} rho_k)

    where rho_k is the sample autocorrelation at lag k and K_max is the
    first lag where the autocorrelation drops below 0.05.

    Parameters
    ----------
    chain : ndarray, shape (M, d)

    Returns
    -------
    ess : ndarray, shape (d,)
    """
    M, d = chain.shape
    ess = np.empty(d)
    for j in range(d):
        x = chain[:, j] - chain[:, j].mean()
        var = np.var(x, ddof=1)
        if var < 1e-14:
            ess[j] = M
            continue
        # Autocorrelation sum
        ac_sum = 0.0
        for lag in range(1, M):
            rho = np.dot(x[:-lag], x[lag:]) / ((M - lag) * var)
            if abs(rho) < 0.05:
                break
            ac_sum += rho
        ess[j] = M / max(1.0, 1.0 + 2.0 * ac_sum)
    return ess


def print_diagnostics(chains: List[np.ndarray], param_names: List[str]) -> None:
    """
    Print R-hat and ESS for each parameter across the given chains.

    Parameters
    ----------
    chains      : list of ndarray, each shape (M, d)
    param_names : list of str, length d
    """
    r_hat = gelman_rubin(chains)
    print("Gelman-Rubin R-hat (target < 1.1):")
    for i, name in enumerate(param_names):
        converged = "OK" if r_hat[i] < 1.1 else "NOT CONVERGED"
        print(f"  {name}: R-hat = {r_hat[i]:.4f}  [{converged}]")

    # ESS from combined chain
    combined = np.concatenate(chains, axis=0)
    ess = effective_sample_size(combined)
    print("Effective Sample Size (combined chains):")
    for i, name in enumerate(param_names):
        print(f"  {name}: ESS = {ess[i]:.1f}")
