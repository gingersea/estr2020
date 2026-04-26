"""
vi.py
-----
Mean-field Stochastic Gradient Variational Inference (SGVI) with the
reparameterization trick.

Variational Family
------------------
  q(theta; lambda) = N(theta; mu, diag(sigma^2))

where lambda = (mu, rho) and sigma_i = exp(rho_i) > 0.
Optimizing over unconstrained rho avoids positivity constraints on sigma.

ELBO
----
  L(lambda) = E_{q}[log p(D|theta) + log p(theta) - log q(theta; lambda)]

Gradient Estimator (Reparameterization Trick)
---------------------------------------------
  theta = mu + sigma * epsilon,   epsilon ~ N(0, I)

  grad_lambda L ≈ (1/M) * sum_{m=1}^{M} grad_lambda [log p(theta^m, D) - log q(theta^m; lambda)]

where theta^m = mu + sigma * epsilon^m.

This estimator has lower variance than score-function (REINFORCE) estimators and
is compatible with automatic differentiation (we implement it manually here).

Optimizer
---------
Adam with step-size η, β1=0.9, β2=0.999, ε=1e-8.
"""

import numpy as np
from typing import Callable, Tuple, List


# ---------------------------------------------------------------------------
# ELBO computation
# ---------------------------------------------------------------------------

def compute_elbo(
    mu: np.ndarray,
    rho: np.ndarray,
    log_joint: Callable[[np.ndarray], float],
    n_mc: int = 64,
    rng: np.random.Generator = None,
) -> float:
    """
    Estimate the ELBO using Monte Carlo sampling with the reparameterization trick.

    L(lambda) ≈ (1/M) * sum_m [ log p(theta^m, D) - log q(theta^m; lambda) ]

    where theta^m = mu + sigma * epsilon^m, epsilon^m ~ N(0, I).

    Parameters
    ----------
    mu       : ndarray, shape (d,) — variational mean
    rho      : ndarray, shape (d,) — log-std parameters (sigma = exp(rho))
    log_joint: callable — log p(theta, D) = log p(theta) + log p(D|theta)
    n_mc     : int — number of MC samples
    rng      : random generator

    Returns
    -------
    float — ELBO estimate
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sigma = np.exp(rho)
    d = mu.shape[0]

    elbo = 0.0
    for _ in range(n_mc):
        eps = rng.standard_normal(d)
        theta = mu + sigma * eps

        # log p(theta, D)
        log_p = log_joint(theta)

        # log q(theta; lambda) = sum_i log N(theta_i; mu_i, sigma_i^2)
        log_q = np.sum(
            -0.5 * np.log(2.0 * np.pi)
            - rho                               # = -log(sigma_i)
            - 0.5 * eps ** 2                    # = -(theta_i - mu_i)^2 / (2*sigma_i^2)
        )

        elbo += log_p - log_q

    return elbo / n_mc


# ---------------------------------------------------------------------------
# Gradient of the ELBO w.r.t. (mu, rho)
# ---------------------------------------------------------------------------

def elbo_gradient(
    mu: np.ndarray,
    rho: np.ndarray,
    log_joint: Callable[[np.ndarray], float],
    n_mc: int = 64,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate grad_{mu, rho} ELBO via the reparameterization trick.

    Derivation (for a single MC sample, then average):
      L = log p(theta, D) - log q(theta; lambda)
      theta = mu + sigma * eps,  sigma = exp(rho)

      dL/d_mu_i   = d/d_mu_i [ log p(mu + sigma*eps, D) - log q(...) ]
                  = grad_theta log p~(theta)_i  +  (theta_i - mu_i) / sigma_i^2
                    Note: d(-log q)/d_mu_i = (theta_i - mu_i)/sigma_i^2

      dL/d_rho_i  = grad_theta log p~(theta)_i * eps_i * sigma_i  (chain rule on theta)
                    + d(-log q)/d_rho_i
      where  d(-log q)/d_rho_i = 1 - eps_i^2   (since log q has term +rho_i + 0.5*eps_i^2)

    We compute gradients of log p using finite differences (central difference)
    since we do not have autodiff.

    Parameters
    ----------
    mu, rho    : ndarray, shape (d,)
    log_joint  : callable
    n_mc       : int
    rng        : random generator

    Returns
    -------
    grad_mu  : ndarray, shape (d,)
    grad_rho : ndarray, shape (d,)
    elbo_est : float
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sigma = np.exp(rho)
    d = mu.shape[0]
    h = 1e-5  # finite-difference step

    grad_mu  = np.zeros(d)
    grad_rho = np.zeros(d)
    elbo_acc = 0.0

    for _ in range(n_mc):
        eps = rng.standard_normal(d)
        theta = mu + sigma * eps

        # Gradient of log_joint w.r.t. theta via central differences
        grad_log_p = np.empty(d)
        log_p = log_joint(theta)
        for i in range(d):
            theta_p = theta.copy(); theta_p[i] += h
            theta_m = theta.copy(); theta_m[i] -= h
            grad_log_p[i] = (log_joint(theta_p) - log_joint(theta_m)) / (2.0 * h)

        # log q
        log_q = np.sum(-0.5 * np.log(2.0 * np.pi) - rho - 0.5 * eps ** 2)

        # gradient of (-log q) w.r.t. mu and rho
        # d(-log q)/d_mu_i = (theta_i - mu_i) / sigma_i^2 = eps_i / sigma_i
        d_neg_logq_dmu = eps / sigma
        # d(-log q)/d_rho_i = 1 - eps_i^2
        d_neg_logq_drho = 1.0 - eps ** 2

        # grad ELBO via chain rule (reparameterization):
        # d_mu:  grad_theta log p * d_theta/d_mu + d(-log q)/d_mu
        #        d_theta/d_mu = I  (identity)
        grad_mu  += grad_log_p + d_neg_logq_dmu

        # d_rho: grad_theta log p * d_theta/d_rho + d(-log q)/d_rho
        #        d_theta/d_rho_i = eps_i * sigma_i (chain rule: sigma=exp(rho))
        grad_rho += grad_log_p * eps * sigma + d_neg_logq_drho

        elbo_acc += log_p - log_q

    return grad_mu / n_mc, grad_rho / n_mc, elbo_acc / n_mc


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

class Adam:
    """Adaptive moment estimation (Adam) optimizer."""

    def __init__(self, lr: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: dict = {}
        self.v: dict = {}

    def step(self, params: np.ndarray, grads: np.ndarray, key: str) -> np.ndarray:
        """
        Perform one Adam update (gradient ascent: params += update).

        Parameters
        ----------
        params : ndarray
        grads  : ndarray — gradient of objective to MAXIMIZE
        key    : str — parameter identifier (to maintain separate moments)

        Returns
        -------
        updated params : ndarray
        """
        self.t += 1
        if key not in self.m:
            self.m[key] = np.zeros_like(params)
            self.v[key] = np.zeros_like(params)

        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * grads ** 2

        m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)

        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Main VI training loop
# ---------------------------------------------------------------------------

def run_vi(
    log_joint: Callable[[np.ndarray], float],
    d: int,
    n_iter: int = 5_000,
    lr: float = 0.01,
    n_mc: int = 64,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Run mean-field stochastic gradient VI.

    Parameters
    ----------
    log_joint : callable — log p(theta, D) = log_prior(theta) + log_likelihood(theta, data)
    d         : int — parameter dimension
    n_iter    : int — number of gradient steps
    lr        : float — Adam learning rate
    n_mc      : int — Monte Carlo samples per gradient estimate
    seed      : int — random seed

    Returns
    -------
    mu_opt   : ndarray, shape (d,) — optimized variational mean
    rho_opt  : ndarray, shape (d,) — optimized log-std
    elbo_history : list of float — ELBO at each iteration
    """
    rng = np.random.default_rng(seed)

    # Initialization: mu=0, rho=0 (sigma=1)
    mu  = np.zeros(d)
    rho = np.zeros(d)

    optimizer = Adam(lr=lr)
    elbo_history: List[float] = []

    for t in range(n_iter):
        g_mu, g_rho, elbo_est = elbo_gradient(mu, rho, log_joint,
                                               n_mc=n_mc, rng=rng)

        # Gradient ASCENT (maximizing ELBO)
        mu  = optimizer.step(mu,  g_mu,  key="mu")
        rho = optimizer.step(rho, g_rho, key="rho")

        elbo_history.append(elbo_est)

        if (t + 1) % 500 == 0:
            print(f"  VI iter {t+1:5d}/{n_iter} | ELBO = {elbo_est:10.4f}")

    return mu, rho, elbo_history


def vi_samples(mu: np.ndarray, rho: np.ndarray,
               n_samples: int = 2000, seed: int = 1) -> np.ndarray:
    """
    Draw samples from the optimized variational posterior q(theta; lambda*).

    Parameters
    ----------
    mu, rho   : ndarray, shape (d,)
    n_samples : int
    seed      : int

    Returns
    -------
    samples : ndarray, shape (n_samples, d)
    """
    rng = np.random.default_rng(seed)
    sigma = np.exp(rho)
    eps = rng.standard_normal((n_samples, len(mu)))
    return mu + sigma * eps
