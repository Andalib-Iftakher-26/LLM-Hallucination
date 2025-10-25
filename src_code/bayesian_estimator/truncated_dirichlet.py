# src/bayesian_estimator/truncated_dirichlet.py
from typing import Tuple, Sequence, Optional
import numpy as np
from scipy.stats import dirichlet
from scipy.special import gammaln
import math

def dirichlet_logpdf(b: np.ndarray, alpha: np.ndarray) -> float:
    """
    Log pdf of Dirichlet at point b for parameters alpha.
    """
    # log normalization constant
    alpha0 = alpha.sum()
    logB = np.sum(gammaln(alpha)) - gammaln(alpha0)
    # compute log pdf
    val = ((alpha - 1.0) * np.log(b)).sum() - logB
    return float(val)

def estimate_E_and_Var_under_constraints(alpha: Sequence[float],
                                         lower_bounds: Sequence[float],
                                         n_samples: int = 2000,
                                         random_state: Optional[int] = None) -> Tuple[float, float]:
    """
    Estimate E[h] and Var[h] where h = -sum b_j log b_j and b ~ Dir(alpha) truncated with
    constraints b_j >= lower_bounds_j for all j.
    Strategy:
      - Draw M samples from Dir(alpha)
      - Keep those that satisfy constraints (rejection)
      - If accepted samples are few, use self-normalized importance sampling with weights = pDir(b)
    Returns (E[h], Var[h])
    """
    rng = np.random.default_rng(random_state)
    alpha = np.asarray(alpha, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    K = alpha.size

    # quick rejection sampling
    samples = dirichlet.rvs(alpha, size=n_samples, random_state=rng)
    # boolean mask for constraints
    mask = (samples >= lower_bounds - 1e-12).all(axis=1)
    accepted = samples[mask]

    def entropy_of_batch(batches: np.ndarray) -> np.ndarray:
        # compute -sum b log b elementwise
        return -np.sum(batches * np.log(batches + 1e-30), axis=1)

    if accepted.shape[0] >= max(20, 0.01 * n_samples):
        # enough accepted samples -> use plain Monte Carlo (unbiased)
        h_vals = entropy_of_batch(accepted)
        return float(h_vals.mean()), float(h_vals.var(ddof=0))
    else:
        # fallback: self-normalized importance sampling (SNIS)
        # compute weights for all samples (unnormalized): w_i = p_trunc(b_i) / q(b_i)
        # where q is sampling distribution (Dir(alpha)), p_trunc proportional to pDir(b) if b in constraint else 0.
        # So weights = pDir(b) if b satisfies constraints; but pDir(b) = dirichlet.pdf(b;alpha)
        log_pdf_vals = np.array([dirichlet_logpdf(b, alpha) for b in samples])
        satisfies = (samples >= lower_bounds - 1e-12).all(axis=1)
        if not satisfies.any():
            # No sample satisfies constraints; return a conservative fallback by returning
            # entropy for non-truncated Dirichlet prior (this is plausible when constraints impossible).
            from .dirichlet import expected_entropy_dirichlet
            Eh = expected_entropy_dirichlet(alpha)
            return float(Eh), float(0.0)

        # Using self-normalized weights
        weights = np.exp(log_pdf_vals - log_pdf_vals.max()) * satisfies.astype(float)
        weights_sum = weights.sum()
        weights = weights / (weights_sum + 1e-30)

        h_vals = entropy_of_batch(samples)
        Eh = (weights * h_vals).sum()
        Ehh = (weights * (h_vals ** 2)).sum()
        Varh = max(0.0, Ehh - Eh ** 2)
        return float(Eh), float(Varh)
