import numpy as np
import math
from scipy.stats import dirichlet

class BayesianSemanticEntropy:
    def __init__(self, alpha=1.0, max_mc_samples=5000):
        self.alpha = alpha  # Hyperparameter for Dirichlet distribution
        self.max_mc_samples = max_mc_samples  # Max number of Monte Carlo samples
        self.lambda_unseen = 1.0  # Expected number of unseen meanings
        self.max_unseen = 4  # Maximum number of unseen meanings to consider


    def _shannon_entropy(self, probability_vector):
        """
        Calculate Shannon entropy for a given probability vector.
        """
        p = np.array(probability_vector) + 1e-12  # Small value to avoid log(0)
        return -np.sum(p * np.log(p))

    def estimate_entropy(self, meaning_clusters):
        if not meaning_clusters:
            return 0.0, 0.0

        # -----------------------------
        # 1) Build lower bounds and counts per meaning from your cluster format
        # -----------------------------
        meaning_prob_mass = {}
        meaning_seq_counts = {}

        for cluster in meaning_clusters:
            m_id = cluster["meaning_id"]
            probs = cluster.get("probabilities", [])

            if probs is None:
                probs = []

            probs = np.array(probs, dtype=float)
            if probs.size == 0:
                continue

            meaning_prob_mass[m_id] = float(np.sum(probs))   # lower bound b_m
            meaning_seq_counts[m_id] = int(len(probs))       # count c_m

        if not meaning_prob_mass:
            return 0.0, 0.0

        unique_meanings = sorted(meaning_prob_mass.keys())
        K_observed = len(unique_meanings)

        lower_bounds = np.array([meaning_prob_mass[m] for m in unique_meanings], dtype=float)
        sum_lb = float(np.sum(lower_bounds))

        # If observed meanings already consume ~all probability mass => deterministic
        eps = 1e-6
        if sum_lb >= 1.0 - eps:
            p = lower_bounds / max(sum_lb, eps)
            return float(self._shannon_entropy(p)), 0.0

        # -----------------------------
        # 2) Distribution over unseen meanings (hierarchical over K)
        #    Truncated Poisson over num_unseen in [0..max_unseen]
        # -----------------------------
        lambda_unseen = float(getattr(self, "lambda_unseen", 1.0))
        max_unseen = int(getattr(self, "max_unseen", 4))

        

        raw_w = np.array([self._poisson_pmf(u, lambda_unseen) for u in range(max_unseen + 1)], dtype=float)
        if np.sum(raw_w) <= 0:
            raw_w = np.ones(max_unseen + 1, dtype=float)
        w_unseen = raw_w / np.sum(raw_w)  # weights over num_unseen

        # Counts vector aligned with unique_meanings
        counts_vec = np.array([meaning_seq_counts[m] for m in unique_meanings], dtype=float)

        means_per_K = []
        vars_per_K = []
        used_weights = []

        # -----------------------------
        # 3) For each possible K = K_observed + num_unseen:
        #    sample Dirichlet, enforce bounds, compute H distribution, store mean/var
        # -----------------------------
        for num_unseen in range(0, max_unseen + 1):
            K_total = K_observed + num_unseen

            alpha_vec = np.full(K_total, float(self.alpha), dtype=float)
            alpha_vec[:K_observed] += counts_vec  # add sequence counts to observed meanings

            try:
                candidate_vectors = dirichlet.rvs(alpha_vec, size=int(self.max_mc_samples))
            except ValueError:
                continue

            observed_candidates = candidate_vectors[:, :K_observed]
            valid_mask = np.all(observed_candidates >= lower_bounds, axis=1)
            valid_vectors = candidate_vectors[valid_mask]

            if len(valid_vectors) < 2:
                continue

            entropies = np.array([self._shannon_entropy(vec) for vec in valid_vectors], dtype=float)

            means_per_K.append(float(np.mean(entropies)))
            vars_per_K.append(float(np.var(entropies)))
            used_weights.append(float(w_unseen[num_unseen]))

        if not means_per_K:
            return 0.0, 0.0

        means_per_K = np.array(means_per_K, dtype=float)
        vars_per_K = np.array(vars_per_K, dtype=float)
        used_weights = np.array(used_weights, dtype=float)
        used_weights = used_weights / np.sum(used_weights)

        # -----------------------------
        # 4) Hierarchical combine over K (law of total expectation/variance)
        # -----------------------------
        final_entropy = float(np.sum(used_weights * means_per_K))
        final_variance = float(np.sum(used_weights * (vars_per_K + means_per_K**2)) - final_entropy**2)

        # numeric safety
        if final_variance < 0 and final_variance > -1e-12:
            final_variance = 0.0

        return final_entropy, final_variance


    def _poisson_pmf(self, k: int, lam: float) -> float:
        # stable pmf in log-space: exp(-lam) * lam^k / k!
        return math.exp(-lam + k * math.log(lam + 1e-30) - math.lgamma(k + 1))
    