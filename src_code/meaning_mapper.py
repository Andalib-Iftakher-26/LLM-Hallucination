import numpy as np
import json
from sklearn.preprocessing import normalize
import gensim.downloader as api
from Load_model_output import create_cleaned_dataset, path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13
import os
from gensim.models import KeyedVectors

SIMILARITY_THRESHOLD = 0.85
VECTOR_DIM = 300

word_vectors = api.load("word2vec-google-news-300")
print("Model loaded successfully.")

def vectorize_sentence(sentence, model):
    words = sentence.lower().split()
    
    word_vecs = []
    for word in words:
        if word in model.key_to_index:
            word_vecs.append(model[word])
        
    
    if word_vecs == []:
        return np.zeros(VECTOR_DIM)
    
    sentence_vec = np.mean(word_vecs, axis=0)
    sentence_vec = np.nan_to_num(sentence_vec)
    
    
    return normalize(sentence_vec.reshape(1, -1))[0]






######################## FOR CHECKING SIMILARITY ###########################
def euclidean_distance(A, B):
    return np.linalg.norm(A - B)

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def manhattan_distance(A, B):
    return np.sum(np.abs(A - B))

def jaccard_similarity(A, B):
    intersection = np.sum(np.minimum(A, B))
    union = np.sum(np.maximum(A, B))
    return intersection / union

def dot_product(A, B):
    return np.dot(A, B)

def pearson_correlation(A, B):
    return np.corrcoef(A, B)[0, 1]

def rbf_kernel(A, B, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(A - B) ** 2)

def dice_coefficient(A, B):
    intersection = np.sum(np.minimum(A, B))
    return 2 * intersection / (np.sum(A) + np.sum(B))





def cluster_responses(list_of_responses, model, threshold):
    response_vectors = [vectorize_sentence(resp, model) for resp in list_of_responses]

    cluster = {}          # {cluster_id: {rep_idx: [member_indices...]}}
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
            sim = cosine_similarity(vec, rep_vec)[0][0]
            if sim >= threshold:
                cluster[j][rep_idx].append(i)
                placed = True
                break

        if not placed:
            cluster[cluster_id] = {i: [i]}
            cluster_id += 1

    # return flat cluster ids aligned to responses
    cluster_ids = [-1] * len(list_of_responses)
    for cid, m in cluster.items():
        rep_idx = next(iter(m.keys()))
        for idx in m[rep_idx]:
            cluster_ids[idx] = cid
    return cluster_ids



if __name__ == "__main__":

    all_paths = [path1] 
    all_paths += [path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13]
    
    for file_path in all_paths:
        print(f"\n========================================================")
        print(f"STARTING PROCESSING FOR FILE: {os.path.basename(file_path)}")
        print(f"========================================================")

        cleaned_data = create_cleaned_dataset(file_path)
        final_output = {}
        
        for i in range(len(cleaned_data.questions)):
            prompt_key = f"prompt_{i}: {cleaned_data.questions[i]}"
            responses = cleaned_data.response_list[i]

            print(f"\n--- Clustering responses for prompt {i} ---")
            assigned_ids = cluster_responses(responses, word_vectors, SIMILARITY_THRESHOLD)

            prompt_clusters = {}

            for j, response_text in enumerate(responses):
                cluster_id = assigned_ids[j]
                if cluster_id not in prompt_clusters:
                    prompt_clusters[cluster_id] = {
                        "meaning_id": cluster_id,
                        "members": []
                    }
                prompt_clusters[cluster_id]["members"].append(response_text)

            final_output[prompt_key] = list(prompt_clusters.values())

        base_file = os.path.basename(file_path)
        output_name = base_file.replace('.pickle', '_clusters.json')
        output_filename = f"D:/LLM-Hallucination/data/meanings/cos_sim/{output_name}"
        with open(output_filename, 'w') as f:
            json.dump(final_output, f, indent=2)

        print(f"\n Final clusters for {base_file} saved to '{output_filename}'")
