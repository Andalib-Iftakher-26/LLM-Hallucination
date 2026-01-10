
# run_adaptive.py
import json
import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from bayesian_estimator import BayesianSemanticEntropy, KPrior

# =========================
# CONFIG
# =========================
CLUSTERS_DIRECTORY_COSINE_SIM = r"D:/LLM HALL/LLM-Hallucination/data/meanings/cosine_sim"
CLUSTERS_DIRECTORY_PEARSON_SIM = r"D:/LLM HALL/LLM-Hallucination/data/meanings/pearson_sim"
CLUSTERS_DIRECTORY_RBF_SIM = r"D:/LLM HALL/LLM-Hallucination/data/meanings/rbf_sim"
AUTHOR = r"D:\LLM HALL\LLM-Hallucination\data\Authors"

OUTPUT_DIRECTORY = r"D:/LLM HALL/LLM-Hallucination/result"

# Adaptive stopping rule: stop once Var[h] is below threshold (higher precision).
VARIANCE_THRESHOLD = 0.005

# Budget
MAX_SAMPLES = 500

# Random seed
RANDOM_SEED = 0

# K prior learning (paper uses training prompts to learn support-size prior). [1](https://mqoutlook-my.sharepoint.com/personal/andalib_iftakher_students_mq_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/bayesian_estimator.py)
TRAIN_PROMPTS_PER_METRIC = 200
K_SMOOTHING = 1.0
MC_SAMPLES = 8000
ALPHA_DIRICHLET = 1.0


# =========================
# HELPERS
# =========================
def build_pool_from_prompt_clusters(prompt_clusters: List[dict], max_samples: int = MAX_SAMPLES) -> List[dict]:
    """
    prompt_clusters look like:
      [{"meaning_id": 1, "members": [...], "probabilities": [...]}, ...]

    Returns a flat list of samples:
      [{"meaning_id": m, "probability": p, "text": s}, ...]
    """
    pool = []
    for cluster in prompt_clusters or []:
        m_id = int(cluster.get("meaning_id"))
        probs = cluster.get("probabilities", []) or []
        members = cluster.get("members", []) or []

        for i, p in enumerate(probs):
            try:
                pf = float(p)
            except Exception:
                continue

            pool.append({
                "meaning_id": m_id,
                "probability": pf,
                "text": members[i] if i < len(members) else ""
            })
            if len(pool) >= max_samples:
                return pool
    return pool


def samples_to_clusters_with_text(samples: List[dict]) -> List[dict]:
    """
    Convert flat samples into estimator input:
      [{"meaning_id": mid, "members": [...], "probabilities": [...]}, ...]
    """
    by_mid_members = defaultdict(list)
    by_mid_probs = defaultdict(list)

    for s in samples:
        mid = int(s["meaning_id"])
        by_mid_members[mid].append(s.get("text", ""))
        by_mid_probs[mid].append(float(s["probability"]))

    clusters = []
    for mid in sorted(by_mid_probs.keys()):
        clusters.append({
            "meaning_id": mid,
            "members": by_mid_members[mid],
            "probabilities": by_mid_probs[mid]
        })
    return clusters


def compute_support_size_from_prompt(prompt_data: dict) -> int:
    """
    Support size K for a prompt = number of distinct meaning_ids present.
    This is what the paper uses to learn a prior over K (Eq. 5).
    """
    clusters = prompt_data.get("clusters", []) or []
    mids = set()
    for cl in clusters:
        try:
            mids.add(int(cl.get("meaning_id")))
        except Exception:
            continue
    return max(1, len(mids))


def build_k_prior_for_metric(directory_path: str, train_prompts: int = TRAIN_PROMPTS_PER_METRIC) -> KPrior:
    """
    Build a discrete prior over K from the first `train_prompts` prompts encountered in the metric directory.
    This approximates the paper's training-set procedure for Eq. 5.
    """
    support_sizes = []
    collected = 0

    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue

        fp = os.path.join(directory_path, filename)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        for _, prompt_data in data.items():
            support_sizes.append(compute_support_size_from_prompt(prompt_data))
            collected += 1
            if collected >= train_prompts:
                break
        if collected >= train_prompts:
            break

    # Fit prior via estimator helper (histogram + smoothing)
    est = BayesianSemanticEntropy(alpha=ALPHA_DIRICHLET, mc_samples=MC_SAMPLES, k_smoothing=K_SMOOTHING)
    prior = est.fit_k_prior_from_support_sizes(support_sizes)
    return prior


# =========================
# WORKER
# =========================
def process_file(file_path: str, metric_name: str, k_prior: KPrior) -> Tuple[str, Dict]:
    """
    Runs adaptive estimation for all prompts in ONE json file.
    Estimator created inside process (safe for multiprocessing).
    """
    results = {}
    rng = np.random.default_rng(RANDOM_SEED)

    estimator = BayesianSemanticEntropy(
        alpha=ALPHA_DIRICHLET,
        mc_samples=MC_SAMPLES,
        k_prior=k_prior,
        k_smoothing=K_SMOOTHING,
    )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"[{metric_name}] Unexpected data structure: {type(data)} in {file_path}")
            return metric_name, {}

        print(f"[{metric_name}] Running on {os.path.basename(file_path)} ({len(data)} prompts)")

        for prompt_key, prompt_data in data.items():
            p_false_i = prompt_data.get("p_false", None)
            is_hall_i = prompt_data.get("is_hallucination", None)
            clusters = prompt_data.get("clusters", []) or []

            pool_of_samples = build_pool_from_prompt_clusters(clusters, max_samples=MAX_SAMPLES)
            if not pool_of_samples:
                continue

            rng.shuffle(pool_of_samples)

            current_samples = []
            final_entropy, final_var, final_N = 0.0, 0.0, 0

            for n in range(len(pool_of_samples)):
                current_samples.append(pool_of_samples[n])

                clusters_for_estimator = samples_to_clusters_with_text(current_samples)
                entropy, variance = estimator.estimate_entropy(clusters_for_estimator)

                final_entropy = float(entropy)
                final_var = float(variance)
                final_N = n + 1

                # Adaptive stop: once variance is small enough, we stop
                if n >= 1 and final_var < VARIANCE_THRESHOLD:
                    break

            results[prompt_key] = {
                "entropy": final_entropy,
                "variance": final_var,
                "samples_used": final_N,
                "p_false": p_false_i,
                "is_hallucination": is_hall_i,
            }

    except Exception as e:
        print(f"[{metric_name}] Error processing {file_path}: {e}")

    return metric_name, results


# =========================
# MAIN
# =========================
def run_adaptive_experiment():
    directories = [
        (CLUSTERS_DIRECTORY_COSINE_SIM, "cosine_sim"),
        (CLUSTERS_DIRECTORY_PEARSON_SIM, "pearson_sim"),
        (CLUSTERS_DIRECTORY_RBF_SIM, "rbf_sim"),
        (AUTHOR, "author")
    ]

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Precompute K priors per metric (from training prompts) in main process
    k_priors = {}
    for directory_path, metric_name in directories:
        if not os.path.isdir(directory_path):
            print(f"WARNING: Directory not found: {directory_path}")
            continue
        print(f"Fitting K prior for {metric_name} from: {directory_path}")
        k_priors[metric_name] = build_k_prior_for_metric(directory_path, TRAIN_PROMPTS_PER_METRIC)
        # Save the learned prior for reproducibility
        prior_out = os.path.join(OUTPUT_DIRECTORY, f"k_prior_{metric_name}.json")
        with open(prior_out, "w", encoding="utf-8") as f:
            json.dump(k_priors[metric_name].probs, f, indent=2, ensure_ascii=False)
        print(f"Saved K prior: {prior_out}")

    all_results_by_metric = {m: {} for _, m in directories}
    futures = []

    with ProcessPoolExecutor() as executor:
        for directory_path, metric_name in directories:
            if not os.path.isdir(directory_path):
                continue

            k_prior = k_priors.get(metric_name, KPrior({1: 1.0}))
            print(f"Processing files in {metric_name} directory...")

            for filename in os.listdir(directory_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(directory_path, filename)
                    futures.append(executor.submit(process_file, file_path, metric_name, k_prior))

        for future in as_completed(futures):
            metric_name, results = future.result()
            all_results_by_metric.setdefault(metric_name, {}).update(results)

    # Write one file per metric
    for metric_name, metric_results in all_results_by_metric.items():
        if not metric_results:
            print(f"Skipping {metric_name}: no results.")
            continue

        output_filename = f"adaptive_results_{metric_name}.json"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metric_results, f, indent=2, ensure_ascii=False)

        print(f"Saved {metric_name} results to: {output_path}")


if __name__ == "__main__":
    run_adaptive_experiment()
