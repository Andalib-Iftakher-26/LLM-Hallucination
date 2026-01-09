import numpy as np
import json
import os
import gensim.downloader as api
from sklearn.preprocessing import normalize

from Load_model_output import (
    path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13,
    create_cleaned_dataset
)

SIMILARITY_THRESHOLD = 0.85
VECTOR_DIM = 300

# =========================
# Load pre-trained Word2Vec model
# =========================
word_vectors = api.load("word2vec-google-news-300")
print("Model loaded successfully.")

# =========================
# Vectorize a sentence
# =========================
def vectorize_sentence(sentence, model):
    words = str(sentence).lower().split()
    word_vecs = [model[word] for word in words if word in model.key_to_index]

    if not word_vecs:
        return np.zeros(VECTOR_DIM, dtype=float)

    sentence_vec = np.mean(word_vecs, axis=0)
    sentence_vec = np.nan_to_num(sentence_vec)
    return normalize(sentence_vec.reshape(1, -1))[0]

# =========================
# Similarity functions
# =========================
def cosine_similarity(A, B, eps=1e-12):
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    if norm_A < eps and norm_B < eps:
        return 1.0
    if norm_A < eps or norm_B < eps:
        return 0.0

    return float(np.dot(A, B) / (norm_A * norm_B))


def pearson_correlation(A, B):
    mean_A = np.mean(A)
    mean_B = np.mean(B)

    A_centered = A - mean_A
    B_centered = B - mean_B

    stddev_A = np.std(A_centered)
    stddev_B = np.std(B_centered)

    if stddev_A == 0 and stddev_B == 0:
        return 1.0
    elif stddev_A == 0 or stddev_B == 0:
        return 0.0

    return float(np.dot(A_centered, B_centered) / (stddev_A * stddev_B))


def rbf_kernel(A, B, gamma=0.1):
    return float(np.exp(-gamma * np.linalg.norm(A - B) ** 2))

# =========================
# Cluster responses based on similarity
# =========================
def cluster_responses(list_of_responses, model, threshold, similarity_metric):
    response_vectors = [vectorize_sentence(resp, model) for resp in list_of_responses]

    cluster = {}   # {cluster_id: {rep_idx: [member_indices...]}}
    cluster_id = 1

    for i, vec in enumerate(response_vectors):
        if not cluster:
            cluster[cluster_id] = {i: [i]}
            cluster_id += 1
            continue

        placed = False
        for cid in list(cluster.keys()):
            rep_idx = next(iter(cluster[cid].keys()))
            rep_vec = response_vectors[rep_idx]

            if similarity_metric == "cosine":
                sim = cosine_similarity(vec, rep_vec)
            elif similarity_metric == "pearson":
                sim = pearson_correlation(vec, rep_vec)
            elif similarity_metric == "rbf":
                sim = rbf_kernel(vec, rep_vec)
            else:
                raise ValueError(f"Unknown similarity_metric: {similarity_metric}")

            if sim >= threshold:
                cluster[cid][rep_idx].append(i)
                placed = True
                break

        if not placed:
            cluster[cluster_id] = {i: [i]}
            cluster_id += 1

    # Assign cluster ids for each response (same order as responses)
    cluster_ids = [-1] * len(list_of_responses)
    for cid, m in cluster.items():
        rep_idx = next(iter(m.keys()))
        for idx in m[rep_idx]:
            cluster_ids[idx] = cid

    return cluster_ids

# =========================
# Safe prompt-wise extraction
# =========================
def get_prompt_value(obj, i, default=None):
    """
    Gets obj[i] if it exists and is indexable, else default.
    """
    if obj is None:
        return default
    try:
        return obj[i]
    except Exception:
        return default

# =========================
# Main
# =========================
if __name__ == "__main__":
    similarity_metrics = ["cosine", "pearson", "rbf"]

    # Create output directories
    for metric in similarity_metrics:
        output_dir = f"D:/LLM HALL/LLM-Hallucination/data/meanings/{metric}_sim"
        os.makedirs(output_dir, exist_ok=True)

    all_paths = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13]

    for file_path in all_paths:
        print(f"\n========================================================")
        print(f"STARTING PROCESSING FOR FILE: {os.path.basename(file_path)}")
        print(f"========================================================")

        cleaned_data = create_cleaned_dataset(file_path)

        # Loop metrics first
        for metric in similarity_metrics:
            print(f"\nProcessing metric: {metric}...")

            final_output = {}

            # Loop prompts/questions
            for i in range(len(cleaned_data.questions)):
                prompt_key = f"prompt_{i}: {cleaned_data.questions[i]}"

                responses = cleaned_data.response_list[i]
                log_probs_list = cleaned_data.token_log_probs[i]

                #prompt-wise labels stored once per prompt
                p_false_i = None
                is_hall_i = None

                if hasattr(cleaned_data, "p_false"):
                    p_false_i = get_prompt_value(cleaned_data.p_false, i, None)
                    p_false_i = float(p_false_i) if p_false_i is not None else None

                if hasattr(cleaned_data, "is_hallucination"):
                    is_hall_i = get_prompt_value(cleaned_data.is_hallucination, i, None)
                    is_hall_i = bool(is_hall_i) if is_hall_i is not None else None

                # Run clustering (preserves response order inside each cluster)
                assigned_ids = cluster_responses(responses, word_vectors, SIMILARITY_THRESHOLD, metric)

                prompt_clusters = {}
                n_resp = len(responses)

                for j in range(n_resp):
                    cluster_id = int(assigned_ids[j])

                    # sequence probability (preserves response j mapping)
                    seq_prob = float(np.exp(np.sum(log_probs_list[j])))

                    if cluster_id not in prompt_clusters:
                        prompt_clusters[cluster_id] = {
                            "meaning_id": cluster_id,
                            "members": [],
                            "probabilities": []
                        }

                    # ✅ preserves sequence: append in original response order
                    prompt_clusters[cluster_id]["members"].append(responses[j])
                    prompt_clusters[cluster_id]["probabilities"].append(seq_prob)

                # Keep cluster list in insertion order (first-seen cluster order)
                clusters_list = list(prompt_clusters.values())

                # ✅ output format: prompt-wise meta + clusters list
                final_output[prompt_key] = {
                    "p_false": p_false_i,
                    "is_hallucination": is_hall_i,
                    "clusters": clusters_list
                }

            # Save one file per metric
            base_file = os.path.basename(file_path)
            output_name = base_file.replace(".pickle", f"_{metric}_clusters.json")
            output_filename = f"D:/LLM HALL/LLM-Hallucination/data/meanings/{metric}_sim/{output_name}"

            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)

            print(f"Saved {metric} clusters to '{output_filename}'")
