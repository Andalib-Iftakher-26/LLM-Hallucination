import json
import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from bayesian_estimator import BayesianSemanticEntropy

# =========================
# CONFIG
# =========================
CLUSTERS_DIRECTORY_COSINE_SIM = r"D:/LLM HALL/LLM-Hallucination/data/meanings/cosine_sim"
CLUSTERS_DIRECTORY_PEARSON_SIM = r"D:/LLM HALL/LLM-Hallucination/data/meanings/pearson_sim"
CLUSTERS_DIRECTORY_RBF_SIM = r"D:/LLM HALL/LLM-Hallucination/data/meanings/rbf_sim"

OUTPUT_DIRECTORY = r"D:/LLM HALL/LLM-Hallucination/result"
OUTPUT_FILENAME = "adaptive_results.json"

VARIANCE_THRESHOLD = 0.005
MAX_SAMPLES = 500
RANDOM_SEED = 0


# =========================
# HELPERS
# =========================
def samples_to_clusters(samples):
    """
    Convert flat samples:
        [{"meaning_id": m, "probability": p, ...}, ...]
    into estimator input format:
        [{"meaning_id": m, "probabilities": [p1, p2, ...]}, ...]
    """
    by_meaning = defaultdict(list)
    for s in samples:
        by_meaning[int(s["meaning_id"])].append(float(s["probability"]))

    # Keep stable ordering
    return [{"meaning_id": mid, "probabilities": probs}
            for mid, probs in sorted(by_meaning.items(), key=lambda x: x[0])]


def build_pool_from_prompt_clusters(prompt_clusters, max_samples=MAX_SAMPLES):
    """
    prompt_clusters (from your JSON) look like:
      [{"meaning_id": 1, "members": [...], "probabilities": [...]}, ...]

    Returns a flat list of samples.
    """
    pool = []
    for cluster in prompt_clusters:
        m_id = int(cluster.get("meaning_id"))
        probs = cluster.get("probabilities", []) or []
        members = cluster.get("members", []) or []

        for i, p in enumerate(probs):
            pool.append({
                "meaning_id": m_id,
                "probability": float(p),
                "text": members[i] if i < len(members) else ""
            })
            if len(pool) >= max_samples:
                return pool

    return pool


# =========================
# WORKER
# =========================
def process_file(file_path, metric_name):
    """
    Runs adaptive estimation for all prompts in ONE json file.
    IMPORTANT: estimator is created INSIDE the process (safe for multiprocessing).
    """
    results = {}

    estimator = BayesianSemanticEntropy(alpha=1.0)
    rng = np.random.default_rng(RANDOM_SEED)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Debugging: print the structure of data
        print(f"Data is a {type(data)}")
        print(f"First element in data: {list(data.keys())[0] if isinstance(data, dict) else 'No data'}")

        # Check if data is a dictionary (as expected)
        if isinstance(data, dict):
            print(f"Data is a dictionary with {len(data)} prompts.")
        else:
            print(f"Unexpected data structure: {type(data)}")
            return metric_name, {}

        print(f"[{metric_name}] Running on {os.path.basename(file_path)} ({len(data)} prompts)")

        # Process each prompt
        for prompt_key, prompt_data in data.items():
            # Extract metadata (p_false, is_hallucination)
            p_false_i = prompt_data.get("p_false", None)
            is_hall_i = prompt_data.get("is_hallucination", None)
            clusters = prompt_data.get('clusters', [])

            pool_of_samples = build_pool_from_prompt_clusters(clusters, max_samples=MAX_SAMPLES)
            if not pool_of_samples:
                continue

            # Shuffle to avoid file-order bias
            rng.shuffle(pool_of_samples)

            current_samples = []
            final_entropy = 0.0
            final_var = 0.0
            final_N = 0

            for n in range(len(pool_of_samples)):
                current_samples.append(pool_of_samples[n])

                # Convert samples -> clusters before calling estimator
                clusters_for_estimator = samples_to_clusters(current_samples)

                entropy, variance = estimator.estimate_entropy(clusters_for_estimator)

                final_entropy = float(entropy)
                final_var = float(variance)
                final_N = n + 1

                if n >= 1 and final_var < VARIANCE_THRESHOLD:
                    break

            # Store results per prompt (including p_false and is_hallucination)
            results[prompt_key] = {
                "entropy": final_entropy,
                "variance": final_var,
                "samples_used": final_N,
                "p_false": p_false_i,
                "is_hallucination": is_hall_i
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
    ]

    # Store results separately per metric
    all_results_by_metric = {
        "cosine_sim": {},
        "pearson_sim": {},
        "rbf_sim": {},
    }

    futures = []
    with ProcessPoolExecutor() as executor:
        for directory_path, metric_name in directories:
            print(f"Processing files in {metric_name} directory...")

            if not os.path.isdir(directory_path):
                print(f"WARNING: Directory not found: {directory_path}")
                continue

            for filename in os.listdir(directory_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(directory_path, filename)
                    futures.append(executor.submit(process_file, file_path, metric_name))

        for future in as_completed(futures):
            metric_name, results = future.result()
            # Merge into that metric only
            all_results_by_metric.setdefault(metric_name, {}).update(results)

    # Write one file per metric
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

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
