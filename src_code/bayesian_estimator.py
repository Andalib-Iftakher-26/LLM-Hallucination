
# bayesian_estimator.py
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class KPrior:
    """
    Discrete prior over support size K.
    probs: dict mapping K -> probability (sums to 1).
    """
    probs: Dict[int, float]

    def conditioned_on_ge(self, k_min: int) -> "KPrior":
        filt = {k: p for k, p in self.probs.items() if k >= k_min and p > 0}
        if not filt:
            # Fallback: if nothing survives, put all mass on k_min
            return KPrior({k_min: 1.0})
        s = sum(filt.values())
        return KPrior({k: p / s for k, p in filt.items()})


class BayesianSemanticEntropy:
    """
    Bayesian estimator for semantic entropy with unknown number of meanings K.

    Faithful to:
     - Dirichlet belief over meaning probabilities (paper §3.1)
      - Optional constraint using sequence probabilities (paper §3.2, Eq. 4)
      - Unknown K with a discrete prior and aggregation (paper §3.3, Eq. 5–6)
      - Truncated Dirichlet expectations via self-normalized importance sampling (paper Appendix A.2, Eq. 10)

    Input "meaning_clusters" format (per prompt):
      [
        {"meaning_id": int,
         "members": [str, ...],          # generated sequences (can repeat)
         "probabilities": [float, ...]}  # p(s|x) (or proxy; must align with members list)
      ]
    """

    def __init__(
        self,
        alpha: float = 1.0,
        mc_samples: int = 8000,
        eps: float = 1e-12,
        k_prior: Optional[KPrior] = None,
        k_smoothing: float = 1.0,
    ):
        self.alpha = float(alpha)
        self.mc_samples = int(mc_samples)
        self.eps = float(eps)
        self.k_prior = k_prior  # can be set/fit later
        self.k_smoothing = float(k_smoothing)

    # ----------------------------
    # Public: fit K prior from training prompts
    # ----------------------------
    def fit_k_prior_from_support_sizes(
        self,
        support_sizes: List[int],
        k_max: Optional[int] = None,
    ) -> KPrior:
        """
        Build discrete prior over K using a histogram of observed support sizes (training set),
        matching paper §3.3 (Eq. 5).

        Uses additive smoothing to avoid zeros.
        """
        if not support_sizes:
            self.k_prior = KPrior({1: 1.0})
            return self.k_prior

        support_sizes = [int(k) for k in support_sizes if k >= 1]
        if not support_sizes:
            self.k_prior = KPrior({1: 1.0})
            return self.k_prior

        if k_max is None:
            k_max = max(support_sizes)

        k_max = int(max(1, k_max))
        counts = np.zeros(k_max + 1, dtype=float)

        for k in support_sizes:
            if 1 <= k <= k_max:
                counts[k] += 1.0

        # Additive smoothing across 1..k_max
        counts[1:] += self.k_smoothing
        probs = counts[1:] / counts[1:].sum()

        prior = {k: float(probs[k - 1]) for k in range(1, k_max + 1)}
        self.k_prior = KPrior(prior)
        return self.k_prior

    # ----------------------------
    # Core API: estimate entropy mean/variance for one prompt
    # ----------------------------
    def estimate_entropy(self, meaning_clusters: List[dict]) -> Tuple[float, float]:
        """
        Returns:
          (E[h], Var[h]) where h = H(b), b ~ belief over meaning distribution.
        """
        clusters = meaning_clusters or []
        # Build counts (with repeats) and lower bounds (distinct sequences) for observed meanings
        observed_ids, counts_vec, lower_bounds = self._extract_counts_and_lower_bounds(clusters)
        if len(observed_ids) == 0:
            return 0.0, 0.0

        k_min = len(observed_ids)

        # Prior over K
        if self.k_prior is None:
            # If no learned prior was provided, default to a weak prior concentrated at k_min.
            # (Paper uses a learned discrete prior; you should fit it in run_adaptive.py.) 
            self.k_prior = KPrior({k_min: 1.0})

        cond_prior = self.k_prior.conditioned_on_ge(k_min)

        # For each possible K, compute E[H|K], Var[H|K], then aggregate 
        e_list = []
        v_list = []
        w_list = []

        for K, wK in sorted(cond_prior.probs.items(), key=lambda kv: kv[0]):
            eK, vK = self._estimate_given_K(
                K=K,
                k_observed=k_min,
                counts_vec=counts_vec,
                lower_bounds=lower_bounds
            )
            e_list.append(eK)
            v_list.append(vK)
            w_list.append(wK)

        w = np.asarray(w_list, dtype=float)
        w = w / max(w.sum(), self.eps)
        e = np.asarray(e_list, dtype=float)
        v = np.asarray(v_list, dtype=float)

        # Total expectation/variance (paper Eq. 6).
        E = float(np.sum(w * e))
        Var = float(np.sum(w * (v + e * e)) - E * E)
        if Var < 0 and Var > -1e-12:
            Var = 0.0
        return E, Var

    # ----------------------------
    # Internals
    # ----------------------------
    def _extract_counts_and_lower_bounds(
        self, clusters: List[dict]
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """
        counts_vec: counts per meaning (repeats allowed)
        lower_bounds: sum of probabilities of DISTINCT sequences per meaning 
        """
        observed_ids = []
        counts = []
        lbs = []

        for cl in clusters:
            m_id = int(cl.get("meaning_id"))
            members = cl.get("members", []) or []
            probs = cl.get("probabilities", []) or []

            if len(probs) == 0:
                continue

            # Counts use all samples (repeats allowed) (paper defines D can repeat)
            c = int(len(probs))

            # Lower bound uses DISTINCT sequences s within D for that meaning (Eq. 4)
            # If duplicates exist, keep max(prob) for that text (conservative bound).
            seen = {}
            for i, p in enumerate(probs):
                try:
                    p = float(p)
                except Exception:
                    continue
                s = members[i] if i < len(members) else ""
                if s in seen:
                    seen[s] = max(seen[s], p)
                else:
                    seen[s] = p

            lb = float(sum(seen.values()))

            observed_ids.append(m_id)
            counts.append(c)
            lbs.append(lb)

        # Stable ordering by meaning_id
        order = np.argsort(np.asarray(observed_ids, dtype=int))
        observed_ids = [observed_ids[i] for i in order]
        counts_vec = np.asarray([counts[i] for i in order], dtype=float)
        lower_bounds = np.asarray([lbs[i] for i in order], dtype=float)

        # If lower bounds sum is >= 1 (can happen due to probability bugs / approximations),
        # rescale gently so truncated simplex is feasible
        sLB = float(lower_bounds.sum())
        if sLB >= 1.0 - 1e-10:
            lower_bounds = lower_bounds / max(sLB, self.eps) * (1.0 - 1e-10)

        return observed_ids, counts_vec, lower_bounds

    def _estimate_given_K(
        self,
        K: int,
        k_observed: int,
        counts_vec: np.ndarray,
        lower_bounds: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate E[H] and Var(H) for a fixed K via truncated Dirichlet SNIS.
        """
        K = int(K)
        k_observed = int(k_observed)
        if K < k_observed:
            return 0.0, 0.0

        # Dirichlet parameters: alpha + counts for observed, alpha for unseen
        alpha_vec = np.full(K, self.alpha, dtype=float)
        alpha_vec[:k_observed] += counts_vec

        # Build full lower-bound vector (observed LBs, unseen zeros)
        L = np.zeros(K, dtype=float)
        L[:k_observed] = lower_bounds
        sL = float(L.sum())
        if sL >= 1.0 - 1e-10:
            # Essentially fixed distribution
            p = L / max(sL, self.eps)
            H = self._entropy(p)
            return H, 0.0

        # Sample uniformly from truncated simplex:
        R = 1.0 - sL
        rng = np.random.default_rng()
        u = rng.dirichlet(np.ones(K), size=self.mc_samples)
        b = L[None, :] + R * u

        # SNIS weights proportional to Dirichlet in paper
        logw = self._dirichlet_logpdf(b, alpha_vec)
        logw -= np.max(logw)  # stabilize
        w = np.exp(logw)
        w_sum = float(np.sum(w)) + self.eps
        w = w / w_sum

        # Weighted moments of entropy
        ent = self._entropy_batch(b)
        Eh = float(np.sum(w * ent))
        Eh2 = float(np.sum(w * (ent ** 2)))
        Varh = float(max(0.0, Eh2 - Eh * Eh))
        return Eh, Varh

    def _entropy(self, p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float) + self.eps
        p = p / max(float(p.sum()), self.eps)
        return float(-np.sum(p * np.log(p)))

    def _entropy_batch(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float) + self.eps
        P = P / np.clip(P.sum(axis=1, keepdims=True), self.eps, None)
        return -np.sum(P * np.log(P), axis=1)

    def _dirichlet_logpdf(self, X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        log Dirichlet pdf for each row of X.
        """
        X = np.asarray(X, dtype=float)
        alpha = np.asarray(alpha, dtype=float)

        # log B(alpha) = sum lgamma(alpha_i) - lgamma(sum alpha_i)
        logB = np.sum([math.lgamma(float(a)) for a in alpha]) - math.lgamma(float(np.sum(alpha)))
        return -logB + np.sum((alpha - 1.0) * np.log(np.clip(X, self.eps, None)), axis=1)
