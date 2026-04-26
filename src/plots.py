"""
plots.py
--------
Visualization utilities for the Bayesian approximate inference project.

Produces figures saved to the `figures/` directory:
  - trace_plots.pdf     : MCMC trace plots for mu and rho
  - elbo_curve.pdf      : VI ELBO convergence
  - posterior_compare.pdf : MCMC histogram vs VI approximation (per dimension)
  - predictive_compare.pdf: Posterior predictive density comparison
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List

# Allow importing model when plots.py is run from src/
sys.path.insert(0, os.path.dirname(__file__))
from model import mc_predictive

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PARAM_NAMES = [r"$\mu$", r"$\rho = \log\sigma$"]
PARAM_KEYS  = ["mu", "rho"]


def save(name: str):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------

def plot_trace(chains: List[np.ndarray], burn_in: int = 5_000):
    """Trace plots for each chain and parameter (post-burn-in portion)."""
    n_chains = len(chains)
    d = chains[0].shape[1]
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_chains))

    fig, axes = plt.subplots(d, 1, figsize=(10, 3 * d), sharex=False)
    if d == 1:
        axes = [axes]

    for j in range(d):
        ax = axes[j]
        for k, chain in enumerate(chains):
            post_burn = chain[burn_in:]
            ax.plot(post_burn[:, j], alpha=0.6, linewidth=0.5,
                    color=colors[k], label=f"Chain {k+1}")
        ax.set_xlabel("Iteration (post burn-in)")
        ax.set_ylabel(PARAM_NAMES[j])
        ax.set_title(f"Trace plot — {PARAM_NAMES[j]}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save("trace_plots.pdf")


def plot_elbo(elbo_history: List[float]):
    """ELBO convergence curve."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(elbo_history, linewidth=0.8, color="steelblue")
    ax.set_xlabel("VI Iteration")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO Convergence (Stochastic Gradient VI)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save("elbo_curve.pdf")


def plot_posterior_compare(
    mcmc_samples: np.ndarray,
    vi_mu: np.ndarray,
    vi_sigma: np.ndarray,
    true_theta: np.ndarray = None,
):
    """
    For each parameter dimension, overlay:
      - MCMC histogram (normalized)
      - VI Gaussian density N(mu_i, sigma_i^2)
      - True parameter value (dashed vertical line)
    """
    d = mcmc_samples.shape[1]
    fig, axes = plt.subplots(1, d, figsize=(6 * d, 4))
    if d == 1:
        axes = [axes]

    for j in range(d):
        ax = axes[j]
        samples_j = mcmc_samples[:, j]

        # MCMC histogram
        ax.hist(samples_j, bins=50, density=True, alpha=0.5,
                color="steelblue", label="MCMC (MH)")

        # VI Gaussian curve
        x_range = np.linspace(samples_j.min() - 1.0, samples_j.max() + 1.0, 300)
        vi_pdf = (
            1.0 / (np.sqrt(2.0 * np.pi) * vi_sigma[j])
            * np.exp(-0.5 * (x_range - vi_mu[j]) ** 2 / vi_sigma[j] ** 2)
        )
        ax.plot(x_range, vi_pdf, color="darkorange", linewidth=2.0, label="VI (mean-field)")

        # True value
        if true_theta is not None:
            ax.axvline(true_theta[j], color="red", linestyle="--", linewidth=1.5,
                       label="True value")

        ax.set_xlabel(PARAM_NAMES[j])
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior — {PARAM_NAMES[j]}")
        ax.legend()

    plt.tight_layout()
    save("posterior_compare.pdf")


def plot_predictive_compare(
    data: np.ndarray,
    mcmc_samples: np.ndarray,
    vi_samples: np.ndarray,
    x_grid: np.ndarray = None,
):
    """
    Compare posterior predictive densities: MCMC vs VI, overlaid on data histogram.
    """
    if x_grid is None:
        lo = data.min() - 3.0
        hi = data.max() + 3.0
        x_grid = np.linspace(lo, hi, 200)

    print("  Computing MCMC predictive...")
    mcmc_pred = np.array([mc_predictive(x, mcmc_samples) for x in x_grid])
    print("  Computing VI predictive...")
    vi_pred   = np.array([mc_predictive(x, vi_samples)   for x in x_grid])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=30, density=True, alpha=0.4, color="gray", label="Data")
    ax.plot(x_grid, mcmc_pred, color="steelblue",  linewidth=2.0, label="MCMC predictive")
    ax.plot(x_grid, vi_pred,   color="darkorange", linewidth=2.0, linestyle="--",
            label="VI predictive")
    ax.set_xlabel("$x^*$")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Density")
    ax.legend()
    plt.tight_layout()
    save("predictive_compare.pdf")
