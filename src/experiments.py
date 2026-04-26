"""
experiments.py
--------------
Main experiment runner for the Bayesian approximate inference project.

This script:
  1. Generates the toy dataset.
  2. Runs Metropolis-Hastings MCMC (3 chains for Gelman-Rubin).
  3. Runs mean-field stochastic gradient VI.
  4. Computes and prints convergence diagnostics (R-hat, acceptance rate, ELBO).
  5. Computes evaluation metrics: predictive log-likelihood, 95% CI width, wall-clock time.
  6. Produces all figures (trace plots, ELBO curve, posterior comparison, predictive comparison).

Usage
-----
  python experiments.py

All figures are saved to ../figures/.  A summary table is printed to stdout.
"""

import sys
import os
import time
import numpy as np

# Ensure the src directory is on the path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from data import generate_data, TRUE_MU, TRUE_RHO
from model import log_unnorm_posterior, log_pred_density
from mcmc import run_multiple_chains
from vi import run_vi, vi_samples
from diagnostics import gelman_rubin, print_diagnostics
import plots


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

MCMC_CONFIG = dict(
    n_chains  = 3,
    n_iter    = 20_000,
    burn_in   = 5_000,
    step_size = 0.1,
    thin      = 1,
    seed      = 0,
)

VI_CONFIG = dict(
    n_iter = 5_000,
    lr     = 0.01,
    n_mc   = 64,
    seed   = 0,
)

N_PRED_SAMPLES = 2_000   # samples used for predictive evaluation
N_TEST_POINTS  = 200     # test points for predictive log-likelihood


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def predictive_log_likelihood(samples: np.ndarray, test_data: np.ndarray) -> float:
    """
    Average predictive log-likelihood on held-out test data.

    log p(x_test | D) ≈ log [ (1/S) sum_s p(x_test | theta^(s)) ]
    """
    log_pls = []
    for x in test_data:
        log_preds = np.array([log_pred_density(x, s) for s in samples])
        # log-mean-exp for numerical stability
        max_lp = log_preds.max()
        log_mean = max_lp + np.log(np.mean(np.exp(log_preds - max_lp)))
        log_pls.append(log_mean)
    return float(np.mean(log_pls))


def credible_interval_width(samples: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Per-parameter 95% equal-tailed credible interval widths."""
    lo = np.percentile(samples, 100 * alpha / 2, axis=0)
    hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    return hi - lo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Bayesian Approximate Inference — Experiment Runner")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n[1] Generating toy dataset ...")
    data = generate_data()
    print(f"    N={len(data)}, sample mean={data.mean():.3f}, sample std={data.std():.3f}")
    print(f"    True theta: mu={TRUE_MU:.3f}, rho={TRUE_RHO:.3f}")

    # Held-out test data (same DGP, different seed)
    from data import TRUE_SIGMA
    rng_test = np.random.default_rng(99)
    test_data = rng_test.normal(TRUE_MU, TRUE_SIGMA, size=N_TEST_POINTS)

    d = 2   # theta = (mu, rho)
    true_theta = np.array([TRUE_MU, TRUE_RHO])

    # Bind data to log_target / log_joint
    def log_target(theta):
        return log_unnorm_posterior(theta, data)

    # ------------------------------------------------------------------
    # 2. MCMC
    # ------------------------------------------------------------------
    print("\n[2] Running Metropolis-Hastings MCMC ...")
    t0_mcmc = time.perf_counter()
    all_samples, all_chains, accept_rates = run_multiple_chains(
        log_target=log_target, d=d, **MCMC_CONFIG
    )
    t1_mcmc = time.perf_counter()
    mcmc_time = t1_mcmc - t0_mcmc

    for k, ar in enumerate(accept_rates):
        print(f"    Chain {k+1} acceptance rate: {ar*100:.1f}%")

    print("\n  MCMC Convergence Diagnostics:")
    print_diagnostics(all_samples, [r"mu", r"rho"])

    # Combine chains for downstream use
    mcmc_samples_all = np.concatenate(all_samples, axis=0)
    # Subsample to N_PRED_SAMPLES for predictive evaluation
    rng_sub = np.random.default_rng(7)
    idx = rng_sub.choice(len(mcmc_samples_all), size=min(N_PRED_SAMPLES, len(mcmc_samples_all)), replace=False)
    mcmc_samples = mcmc_samples_all[idx]

    # ------------------------------------------------------------------
    # 3. VI
    # ------------------------------------------------------------------
    print("\n[3] Running Mean-Field Stochastic Gradient VI ...")
    t0_vi = time.perf_counter()
    vi_mu, vi_rho, elbo_history = run_vi(log_joint=log_target, d=d, **VI_CONFIG)
    t1_vi = time.perf_counter()
    vi_time = t1_vi - t0_vi

    vi_sigma = np.exp(vi_rho)
    print(f"\n  VI converged lambda*:")
    print(f"    mu  = [{vi_mu[0]:.4f}, {vi_mu[1]:.4f}]")
    print(f"    sigma = [{vi_sigma[0]:.4f}, {vi_sigma[1]:.4f}]")

    vi_samp = vi_samples(vi_mu, vi_rho, n_samples=N_PRED_SAMPLES, seed=42)

    # ------------------------------------------------------------------
    # 4. Evaluation metrics
    # ------------------------------------------------------------------
    print("\n[4] Computing evaluation metrics ...")

    mcmc_pll = predictive_log_likelihood(mcmc_samples, test_data)
    vi_pll   = predictive_log_likelihood(vi_samp, test_data)

    mcmc_ci_width = credible_interval_width(mcmc_samples)
    vi_ci_width   = credible_interval_width(vi_samp)

    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    Results Summary Table                      │")
    print("  ├────────────────────────────┬──────────────┬──────────────────┤")
    print("  │ Metric                     │     MCMC     │       VI         │")
    print("  ├────────────────────────────┼──────────────┼──────────────────┤")
    print(f"  │ Pred. log-likelihood       │  {mcmc_pll:9.4f}  │  {vi_pll:9.4f}      │")
    print(f"  │ 95% CI width (mu)          │  {mcmc_ci_width[0]:9.4f}  │  {vi_ci_width[0]:9.4f}      │")
    print(f"  │ 95% CI width (rho)         │  {mcmc_ci_width[1]:9.4f}  │  {vi_ci_width[1]:9.4f}      │")
    print(f"  │ Wall-clock time (s)        │  {mcmc_time:9.2f}  │  {vi_time:9.2f}      │")
    print("  └────────────────────────────┴──────────────┴──────────────────┘")

    speedup = mcmc_time / vi_time
    print(f"\n  VI is {speedup:.1f}x faster than MCMC.")

    # ------------------------------------------------------------------
    # 5. Figures
    # ------------------------------------------------------------------
    print("\n[5] Generating figures ...")
    plots.plot_trace(all_chains, burn_in=MCMC_CONFIG["burn_in"])
    plots.plot_elbo(elbo_history)
    plots.plot_posterior_compare(mcmc_samples_all, vi_mu, vi_sigma, true_theta)
    plots.plot_predictive_compare(data, mcmc_samples, vi_samp)

    print("\nDone. All figures saved to ../figures/")


if __name__ == "__main__":
    main()
