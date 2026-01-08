import numpy as np
import math
from scipy.stats import dirichlet


class BayesianSemanticEntropy:
    def __init__(self, alpha=1.0, max_mc_samples=5000):
        self.alpha = float(alpha)
        self.max_mc_samples = int(max_mc_samples)

        # Hierarchical prior over unseen meanings
        self.lambda_unseen = 1.0
        self.max_unseen = 4

        # Safety knobs
        self.eps = 1e-12
        self.lb_clip = 0.999999   # lower bound sum cap (prevents impossible constraints)
        self.min_valid = 200      # minimum valid samples desired after filtering
        self.max_resample_rounds = 8  # keep trying to get enough valid samples


    def _shannon_entropy(self, probability_vector):
        p = np.asarray(probability_vector, dtype=float) + self.eps
        p = p / p.sum()
        return float(-np.sum(p * np.log(p)))


    def estimate_entropy(self, meaning_clusters):
        if not meaning_clusters:
            return 0.0, 0.0

        # -----------------------------
        # 1) Build lower bounds and counts per meaning
        # -----------------------------
        meaning_prob_mass = {}
        meaning_seq_counts = {}

        for cluster in meaning_clusters:
            m_id = int(cluster.get("meaning_id"))
            probs = cluster.get("probabilities", []) or []
            probs = np.asarray(probs, dtype=float)

            if probs.size == 0:
                continue

            # lower bound (sum of per-sequence probabilities for that meaning)
            meaning_prob_mass[m_id] = float(probs.sum())
            # count (number of sequences observed for that meaning)
            meaning_seq_counts[m_id] = int(len(probs))

        if not meaning_prob_mass:
            return 0.0, 0.0

        unique_meanings = sorted(meaning_prob_mass.keys())
        K_observed = len(unique_meanings)

        lower_bounds = np.array([meaning_prob_mass[m] for m in unique_meanings], dtype=float)
        sum_lb = float(lower_bounds.sum())

        # ---- SAFETY: if bounds are not in a probability-mass scale, normalize them ----
        # This prevents the constraint "observed_candidates >= lower_bounds" from becoming impossible.
        if sum_lb > self.lb_clip:
            lower_bounds = lower_bounds / max(sum_lb, self.eps) * self.lb_clip
            sum_lb = float(lower_bounds.sum())

        # If bounds already consume ~all mass, entropy becomes nearly deterministic.
        # But we still compute entropy on normalized bounds.
        if sum_lb >= 1.0 - 1e-6:
            p = lower_bounds / max(sum_lb, self.eps)
            return self._shannon_entropy(p), 0.0

        # -----------------------------
        # 2) Prior over unseen meanings (truncated Poisson)
        # -----------------------------
        lam = float(getattr(self, "lambda_unseen", 1.0))
        max_unseen = int(getattr(self, "max_unseen", 4))

        raw_w = np.array([self._poisson_pmf(u, lam) for u in range(max_unseen + 1)], dtype=float)
        if float(raw_w.sum()) <= 0:
            raw_w = np.ones(max_unseen + 1, dtype=float)
        w_unseen = raw_w / raw_w.sum()

        counts_vec = np.array([meaning_seq_counts[m] for m in unique_meanings], dtype=float)

        means_per_K = []
        vars_per_K = []
        used_weights = []

        # -----------------------------
        # 3) For each K_total, sample Dirichlet, apply bounds, estimate entropy distribution
        # -----------------------------
        for num_unseen in range(0, max_unseen + 1):
            K_total = K_observed + num_unseen

            alpha_vec = np.full(K_total, self.alpha, dtype=float)
            alpha_vec[:K_observed] += counts_vec  # posterior-ish update by counts

            # Rejection sampling to get enough valid vectors
            valid_vectors = None
            total_valid = 0

            for _round in range(self.max_resample_rounds):
                try:
                    candidates = dirichlet.rvs(alpha_vec, size=self.max_mc_samples)
                except ValueError:
                    candidates = None

                if candidates is None or len(candidates) == 0:
                    continue

                obs = candidates[:, :K_observed]
                valid_mask = np.all(obs >= lower_bounds, axis=1)
                vv = candidates[valid_mask]

                if valid_vectors is None:
                    valid_vectors = vv
                else:
                    if len(vv) > 0:
                        valid_vectors = np.vstack([valid_vectors, vv])

                total_valid = 0 if valid_vectors is None else len(valid_vectors)
                if total_valid >= self.min_valid:
                    break

            # If still too few valid vectors, FALL BACK (don’t return 0,0)
            # We relax by ignoring the lower-bound constraint for this K.
            if valid_vectors is None or len(valid_vectors) < 2:
                try:
                    valid_vectors = dirichlet.rvs(alpha_vec, size=self.max_mc_samples)
                except ValueError:
                    continue

            entropies = np.array([self._shannon_entropy(v) for v in valid_vectors], dtype=float)

            means_per_K.append(float(entropies.mean()))
            vars_per_K.append(float(entropies.var()))
            used_weights.append(float(w_unseen[num_unseen]))

        if not means_per_K:
            return 0.0, 0.0

        means_per_K = np.asarray(means_per_K, dtype=float)
        vars_per_K = np.asarray(vars_per_K, dtype=float)
        used_weights = np.asarray(used_weights, dtype=float)
        used_weights = used_weights / used_weights.sum()

        # -----------------------------
        # 4) Hierarchical combine across K (total expectation/variance)
        # -----------------------------
        final_entropy = float(np.sum(used_weights * means_per_K))
        final_variance = float(np.sum(used_weights * (vars_per_K + means_per_K ** 2)) - final_entropy ** 2)

        # numeric safety
        if final_variance < 0 and final_variance > -1e-12:
            final_variance = 0.0

        return final_entropy, final_variance


    def _poisson_pmf(self, k: int, lam: float) -> float:
        # exp(-lam) * lam^k / k! in log space
        return math.exp(-lam + k * math.log(lam + 1e-30) - math.lgamma(k + 1))
