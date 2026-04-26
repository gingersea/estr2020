# ESTR2020 Course Project — Bayesian Approximate Inference

**Group XX**

## Overview

This repository contains the source code for the ESTR2020 course project on
Bayesian approximate inference with analytically intractable posteriors.
We implement and compare two classes of methods:

1. **Metropolis-Hastings MCMC** — random-walk Markov chain sampler
2. **Mean-field Stochastic Gradient Variational Inference** — reparameterization-trick-based optimizer

Both methods are applied to a 2-parameter Gaussian model where the posterior
is analytically intractable due to the log-normal parametrization of the variance.

---

## Repository Structure

```
estr2020/
├── src/
│   ├── model.py         # Prior, likelihood, log unnormalized posterior, predictive density
│   ├── data.py          # Toy dataset generation (N=100, true mu=2.0, sigma=1.5)
│   ├── mcmc.py          # Metropolis-Hastings algorithm (single chain + multi-chain)
│   ├── vi.py            # Mean-field SGVI with reparameterization trick + Adam optimizer
│   ├── diagnostics.py   # Gelman-Rubin R-hat, effective sample size
│   ├── plots.py         # Figure generation utilities
│   └── experiments.py   # Main experiment runner (produces all results and figures)
├── figures/             # Auto-generated figures (created by experiments.py)
├── report/
│   ├── report.tex       # Full LaTeX report source
│   └── report.pdf       # Compiled report (16 pages)
└── README.md
```

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install with:
```bash
pip install numpy matplotlib
```

No GPU or deep learning frameworks are required.

---

## How to Reproduce Results

**1. Run the full experiment pipeline:**

```bash
cd src
python experiments.py
```

This will:
- Generate the toy dataset
- Run 3 independent MH-MCMC chains (20,000 iterations each, 5,000 burn-in)
- Run mean-field SGVI (5,000 iterations, Adam lr=0.01, 64 MC samples)
- Print convergence diagnostics (R-hat, acceptance rates, ESS, ELBO)
- Print the results summary table (predictive log-likelihood, CI widths, timings)
- Save all figures to `../figures/`

Expected runtime: approximately 15–20 minutes on a standard laptop CPU.

**2. Compile the LaTeX report:**

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # run twice for cross-references
```

Requires a standard LaTeX installation (e.g., TeX Live).

---

## Model

| Component | Specification |
|-----------|--------------|
| Parameters | θ = (μ, ρ) ∈ ℝ², where ρ = log σ |
| Prior | θ ~ N(0, I₂) |
| Likelihood | xₙ \| θ ~ N(μ, exp(2ρ)), i.i.d. |
| Posterior | p(θ \| D) ∝ p(θ) p(D \| θ) — analytically intractable |
| True values | μ_true = 2.0, σ_true = 1.5, ρ_true ≈ 0.405 |
| Dataset size | N = 100 |

---

## Key Results

| Metric | MCMC (MH) | VI (mean-field) |
|--------|-----------|-----------------|
| Predictive log-likelihood | −1.842 | −1.837 |
| 95% CI width for μ | 0.460 | 0.211 |
| 95% CI width for ρ | 0.281 | 0.468 |
| Gelman-Rubin R̂ (μ) | 1.0003 | — |
| Acceptance rate | ~50.7% | — |
| Wall-clock time (s) | 0.73 | 14.81 |

---

## Module Descriptions

- **`model.py`**: Defines `log_prior`, `log_likelihood`, `log_unnorm_posterior`,
  and `mc_predictive` for Monte Carlo posterior predictive approximation.

- **`data.py`**: Generates the reproducible toy dataset via `generate_data()`.

- **`mcmc.py`**: Implements `metropolis_hastings()` (single chain) and
  `run_multiple_chains()` (parallel overdispersed chains for Gelman-Rubin).

- **`vi.py`**: Implements `elbo_gradient()` (reparameterization-trick gradient estimator),
  `Adam` optimizer, `run_vi()` (main VI loop), and `vi_samples()`.

- **`diagnostics.py`**: Implements `gelman_rubin()` (per-parameter R-hat),
  `effective_sample_size()`, and `print_diagnostics()`.

- **`plots.py`**: Generates `trace_plots.pdf`, `elbo_curve.pdf`,
  `posterior_compare.pdf`, and `predictive_compare.pdf`.

- **`experiments.py`**: Orchestrates the full pipeline and prints the results table.
