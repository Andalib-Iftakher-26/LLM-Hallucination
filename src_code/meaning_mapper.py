import numpy as np
import warnings
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import gensim.downloader as api




SIMILARITY_THRESHOLD = 0.85 
VECTOR_DIM = 300 # Vector dimension for the chosen model


word_vectors = api.load("glove-wiki-gigaword-100")
print("Model loaded successfully.")




def vectorize_sentence(sentence, model):
    words = sentence.lower().split()
    word_vecs = [model[word] for word in words if word in model.key_to_index]
    
    
    if not word_vecs:
        return np.zeros(VECTOR_DIM)
    
    # Calculate the mean of the word vectors
    sentence_vec = np.mean(word_vecs, axis=0)
    
    # Handle potential NaN values and ensure the vector is normalized (has a length of 1)
    sentence_vec = np.nan_to_num(sentence_vec)
        
    return normalize(sentence_vec.reshape(1, -1))[0]




def cluster_responses(list_of_responses, model, threshold):
    response_vectors = [vectorize_sentence(resp, model) for resp in list_of_responses]

    cluster_ids = [-1] * len(list_of_responses)
    next_cluster_id = 0
    
    for i in range(len(list_of_responses)):
        # If this response has not been assigned to a cluster 
        if cluster_ids[i] == -1:
            # start a new cluster.
            cluster_ids[i] = next_cluster_id
            
            #find all other un-clustered responses
            for j in range(i + 1, len(list_of_responses)):
                if cluster_ids[j] == -1:
                    vec_i = response_vectors[i].reshape(1, -1)
                    vec_j = response_vectors[j].reshape(1, -1)
                    
                    similarity = cosine_similarity(vec_i, vec_j)[0][0]
                    
                    if similarity > threshold:
                        # If  similar, assign the other response to the same cluster
                        cluster_ids[j] = next_cluster_id
            
            # Move to the next cluster ID
            next_cluster_id += 1
            
    return cluster_ids




if __name__ == "__main__":
    all_prompt_data = {}
    
    final_output = {}

    print("\n--- Starting Clustering Process ---")
    for prompt_key, responses in all_prompt_data.items():
        print(f"\nClustering responses for '{prompt_key}':")
        
        
        assigned_ids = cluster_responses(responses, word_vectors, SIMILARITY_THRESHOLD)
        
        # Format the results for this prompt
        prompt_clusters = {}
        for i, response_text in enumerate(responses):
            cluster_id = assigned_ids[i]
            if cluster_id not in prompt_clusters:
                prompt_clusters[cluster_id] = {
                    "meaning_id": cluster_id,
                    "members": []
                }
            prompt_clusters[cluster_id]["members"].append(response_text)
            print(f"  -> Assigning to cluster {cluster_id}: '{response_text}'")
        
        # Store the formatted clusters
        final_output[prompt_key] = list(prompt_clusters.values())

    # --- 5. FINAL OUTPUT: Save clusters to a JSON file ---
    
    output_filename = "word2vec_clusters.json"
    with open(output_filename, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\n--- Final Clusters saved to '{output_filename}' ---")
    print(json.dumps(final_output, indent=2))