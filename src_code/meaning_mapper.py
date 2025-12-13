import numpy as np
import json
import os
import gensim.downloader as api
from sklearn.preprocessing import normalize
from Load_model_output import path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, create_cleaned_dataset


SIMILARITY_THRESHOLD = 0.85
VECTOR_DIM = 300

# Load pre-trained Word2Vec model
word_vectors = api.load("word2vec-google-news-300")
print("Model loaded successfully.")

# Vectorize a sentence
def vectorize_sentence(sentence, model):
    words = sentence.lower().split()
    word_vecs = [model[word] for word in words if word in model.key_to_index]
    
    if not word_vecs:
        return np.zeros(VECTOR_DIM)
    
    sentence_vec = np.mean(word_vecs, axis=0)
    sentence_vec = np.nan_to_num(sentence_vec)
    
    return normalize(sentence_vec.reshape(1, -1))[0]

# Define similarity functions
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def pearson_correlation(A, B):
    return np.corrcoef(A, B)[0, 1]

def rbf_kernel(A, B, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(A - B) ** 2)


# Cluster responses based on similarity
def cluster_responses(list_of_responses, model, threshold, similarity_metric):
    response_vectors = [vectorize_sentence(resp, model) for resp in list_of_responses]
    cluster = {}  # {cluster_id: {rep_idx: [member_indices...]}}
    cluster_id = 1

    for i, vec in enumerate(response_vectors):
        if not cluster:
            cluster[cluster_id] = {i: [i]}
            cluster_id += 1
            continue

        placed = False
        for j in list(cluster.keys()):
            
            rep_idx = next(iter(cluster[j].keys()))
            rep_vec = response_vectors[rep_idx]
            if similarity_metric == 'cosine':
                sim = cosine_similarity(vec, rep_vec)

            
            elif similarity_metric == 'pearson':
                sim = pearson_correlation(vec, rep_vec)
            elif similarity_metric == 'rbf':
                sim = rbf_kernel(vec, rep_vec)

            
            if sim >= threshold:
                cluster[j][rep_idx].append(i)
                placed = True
                break

        if not placed:
            cluster[cluster_id] = {i: [i]}
            cluster_id += 1

    # Assign cluster ids
    cluster_ids = [-1] * len(list_of_responses)
    for cid, m in cluster.items():
        rep_idx = next(iter(m.keys()))
        for idx in m[rep_idx]:
            cluster_ids[idx] = cid

    return cluster_ids

# Save clustering results for each metric in different directories
if __name__ == "__main__":
    similarity_metrics = ['cosine', 'pearson', 'rbf']
    
    # Create output directories
    for metric in similarity_metrics:
        output_dir = f"D:/LLM-Hallucination/data/meanings/{metric}_sim"
        os.makedirs(output_dir, exist_ok=True)

    all_paths = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13]

    for file_path in all_paths:
        print(f"\n========================================================")
        print(f"STARTING PROCESSING FOR FILE: {os.path.basename(file_path)}")
        print(f"========================================================")

        cleaned_data = create_cleaned_dataset(file_path)
        
        # --- CORRECTED LOOP STRUCTURE ---
        # 1. Loop through Metrics FIRST
        for metric in similarity_metrics:
            print(f"\nProcessing metric: {metric}...")
            
            # Initialize a FRESH dictionary for this specific metric
            final_output = {}

            # 2. Loop through Questions
            for i in range(len(cleaned_data.questions)):
                prompt_key = f"prompt_{i}: {cleaned_data.questions[i]}"
                responses = cleaned_data.response_list[i]
                
                # Get list of log-prob lists: [[-0.1, -0.4], [-0.5, ...]]
                log_probs_list = cleaned_data.token_log_probs[i]

                # Run clustering
                assigned_ids = cluster_responses(responses, word_vectors, SIMILARITY_THRESHOLD, metric)

                prompt_clusters = {}
                for j in range(len(responses)):
                    cluster_id = int(assigned_ids[j]) # Ensure it's a standard python int for JSON
                    
                    # Math Check: sum log-probs, then exp to get sequence probability (0.0 - 1.0)
                    seq_prob = float(np.exp(np.sum(log_probs_list[j])))

                    if cluster_id not in prompt_clusters:
                        prompt_clusters[cluster_id] = {
                            "meaning_id": cluster_id, 
                            "members": [], 
                            "probabilities": []
                        }
                    
                    prompt_clusters[cluster_id]["members"].append(responses[j])
                    prompt_clusters[cluster_id]["probabilities"].append(seq_prob)

                final_output[prompt_key] = list(prompt_clusters.values())

            # 3. Save File ONCE per metric (Efficient)
            base_file = os.path.basename(file_path)
            output_name = base_file.replace('.pickle', f'_{metric}_clusters.json')
            output_filename = f"D:/LLM-Hallucination/data/meanings/{metric}_sim/{output_name}"

            with open(output_filename, 'w') as f:
                json.dump(final_output, f, indent=2)

            print(f"Saved {metric} clusters to '{output_filename}'")
