# src/bayesian_estimator/dirichlet_entropy.py
from typing import Sequence, Tuple
import numpy as np
from scipy.special import psi, polygamma

def expected_entropy_dirichlet(alpha: Sequence[float]) -> float:
    """
    Compute E[H[b]] for b ~ Dir(alpha) using the formula:
      E[h] = psi(alpha0 + 1) - sum_i (alpha_i/alpha0) * psi(alpha_i + 1)
    where alpha0 = sum alpha_i.
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha0 = alpha.sum()
    term1 = psi(alpha0 + 1)
    weights = alpha / alpha0
    term2 = (weights * psi(alpha + 1)).sum()
    return float(term1 - term2)

def second_moment_entropy_dirichlet(alpha: Sequence[float]) -> float:
    """
    Compute E[h^2] for b ~ Dir(alpha) using standard expressions (see paper Appendix A).
    Returns E[h^2]. Caller can compute Var[h] = E[h^2] - (E[h])^2.
    """
    alpha = np.asarray(alpha, dtype=float)
    K = alpha.size
    alpha0 = alpha.sum()

    # Precompute psi and trigamma values used in the expression
    psi_vals = psi(alpha + np.array([1.0]*K))
    psi0 = psi(alpha0 + 1)
    # trigamma = polygamma(1, x)
    trigamma_vals = polygamma(1, alpha + 1)
    trigamma0 = polygamma(1, alpha0 + 1)

    # Following expansions from literature (Wolpert & Wolf, Hausser & Strimmer).
    # We'll implement the formula from the paper appendix A (adapted).
    # Compute E[h] first:
    Eh = psi(alpha0 + 1) - (alpha / alpha0 * psi(alpha + 1)).sum()

    # Compute E[h^2] via the expansion used in the appendix.
    # For robust code we compute sums term by term.
    # Terms for i==j (diagonal)
    term_diag = 0.0
    for i in range(K):
        ai = alpha[i]
        # E[b_i^2 (log b_i)^2] terms simplified via Dirichlet moments
        # Use constants from appendix (approx)
        num = ai * (ai + 1)
        denom = alpha0 * (alpha0 + 1)
        # compute inner bracket: trigamma(ai+2)-trigamma(alpha0+2) + (psi(ai+2)-psi(alpha0+2))^2
        inner = polygamma(1, ai + 2) - polygamma(1, alpha0 + 2)
        inner += (psi(ai + 2) - psi(alpha0 + 2)) ** 2
        term_diag += (num / denom) * inner

    # Terms for i != j
    term_off = 0.0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            ai = alpha[i]
            aj = alpha[j]
            num = ai * aj
            denom = alpha0 * (alpha0 + 1)
            inner = -polygamma(1, alpha0 + 2)
            inner += (psi(ai + 1) - psi(alpha0 + 2)) * (psi(aj + 1) - psi(alpha0 + 2))
            term_off += (num / denom) * inner

    Eh2 = (term_diag + term_off)
    # The appendix has a more compact multiplicative factor; in case of implementation differences,
    # the above matches the described derivation structure. For stability, if Eh2 is negative due to
    # numerical error, clip it.
    Eh2 = float(max(Eh2, 0.0))
    return Eh2
