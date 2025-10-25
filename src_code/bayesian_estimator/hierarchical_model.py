# src/bayesian_estimator/hierarchical_model.py
import json
from typing import List, Tuple, Sequence, Dict
import numpy as np
from .dirichlet import expected_entropy_dirichlet
from .truncated_dirichlet import estimate_E_and_Var_under_constraints

def load_prior_K(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def make_alpha_from_counts(counts: Sequence[int], alpha_scalar: float, K: int) -> np.ndarray:
    """
    Given observed counts for seen meanings (counts length = Kmin <= K) and a scalar prior alpha_scalar,
    produce an alpha vector of length K:
      - first Kmin entries = alpha_scalar + counts_i
      - remaining entries = alpha_scalar (unobserved meaning mass)
    """
    counts = list(counts)
    Kmin = len(counts)
    if K < Kmin:
        raise ValueError("K must be >= Kmin")
    alpha = [alpha_scalar + c for c in counts] + [alpha_scalar] * (K - Kmin)
    return np.asarray(alpha, dtype=float)

def aggregate_over_K(counts: Sequence[int],
                     observed_mass_per_meaning: Sequence[float],
                     prior_K_path: str,
                     alpha_scalar: float,
                     use_truncation: bool,
                     num_mc_samples: int = 500,
                     random_state: int = 0) -> Tuple[float, float]:
    """
    Aggregate E[h] and Var[h] over K using the prior distribution over K.

    counts: list of counts for meanings observed in D (length = Kmin)
    observed_mass_per_meaning: observed sum p(s|x) per observed meaning (length Kmin)
    prior_K_path: path to prior_K.json
    alpha_scalar: Dirichlet prior α
    """
    prior = load_prior_K(prior_K_path)
    K_support = prior["K_support"]
    lambdas = np.asarray(prior["lambda"], dtype=float)
    lambdas = lambdas / lambdas.sum()

    E_h_list = []
    Var_h_list = []
    weights = []

    Kmin = len(counts)
    for K, w in zip(K_support, lambdas):
        if K < Kmin:
            continue  # impossible given observed #clusters
        alpha = make_alpha_from_counts(counts, alpha_scalar, K)
        # Create lower_bounds vector: for observed meanings j, lower_bounds[j] = observed_mass_j; for others 0
        lower_bounds = np.zeros_like(alpha)
        lower_bounds[:Kmin] = np.asarray(observed_mass_per_meaning, dtype=float)

        if use_truncation:
            Eh, Varh = estimate_E_and_Var_under_constraints(alpha, lower_bounds, n_samples=num_mc_samples, random_state=random_state)
        else:
            # use analytic Dirichlet expectations (fast)
            Eh = expected_entropy_dirichlet(alpha)
            # approximate Var[h] via numeric Monte Carlo or set placeholder 0 (for speed). Use small MC:
            # We'll run a small importance MC to get E[h^2] via dirichlet draws:
            import numpy as _np
            samples = _np.random.default_rng(random_state).dirichlet(alpha, size=max(200, int(num_mc_samples/5)))
            hvals = -_np.sum(samples * _np.log(samples + 1e-30), axis=1)
            Varh = float(hvals.var(ddof=0))

        E_h_list.append(Eh)
        Var_h_list.append(Varh)
        weights.append(w)

    weights = np.asarray(weights)
    if weights.sum() == 0:
        # fallback uniform over K >= Kmin
        valid_K_idx = [i for i, K in enumerate(K_support) if K >= Kmin]
        weights = np.zeros_like(weights)
        weights[valid_K_idx] = 1.0 / len(valid_K_idx)

    weights = weights / weights.sum()
    E_h_list = np.asarray(E_h_list)
    Var_h_list = np.asarray(Var_h_list)

    E_h = (weights * E_h_list).sum()
    Var_h = (weights * Var_h_list).sum() + ((weights * (E_h_list - E_h) ** 2).sum())
    return float(E_h), float(Var_h)
