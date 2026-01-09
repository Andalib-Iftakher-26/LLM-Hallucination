import numpy as np
import json
import os
from Load_model_output import path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, originalData


# Function to group responses based on their semantic_ids
def group_responses_by_semantics(responses, semantic_ids, log_probs_list):
    prompt_clusters = {}
    for idx, response in enumerate(responses):
        # Check if semantic_ids[idx] is an array or scalar
        if isinstance(semantic_ids[idx], np.ndarray):  # If it's an array
            cluster_id = int(semantic_ids[idx][0])  # Extract scalar value from the array
        else:
            cluster_id = int(semantic_ids[idx])  # Directly use it if it's a scalar

        seq_prob = float(np.exp(np.sum(log_probs_list[idx])))  # Calculate the sequence probability

        # Create the cluster if it doesn't exist
        if cluster_id not in prompt_clusters:
            prompt_clusters[cluster_id] = {
                "meaning_id": cluster_id, 
                "members": [], 
                "probabilities": []
            }
        
        # Append the response and probability to the corresponding cluster
        prompt_clusters[cluster_id]["members"].append(response)
        prompt_clusters[cluster_id]["probabilities"].append(seq_prob)

    return prompt_clusters


if __name__ == "__main__":
    # Similarity metrics are not needed now because we are directly using semantic_ids
    similarity_metrics = ['semantic_id_based']  # Just a placeholder since clustering is already done.

    # Create output directories
    for metric in similarity_metrics:
        output_dir = f"D:/LLM HALL/LLM-Hallucination/data/Authors"
        os.makedirs(output_dir, exist_ok=True)

    all_paths = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13]

    for file_path in all_paths:
        print(f"\n========================================================")
        print(f"STARTING PROCESSING FOR FILE: {os.path.basename(file_path)}")
        print(f"========================================================")

        # Load the cleaned data
        cleaned_data = originalData(file_path)  
        
        # Loop through all questions and responses
        for metric in similarity_metrics:
            print(f"\nProcessing metric: {metric}...")

            final_output = {}

            # Loop through Questions
            for i in range(len(cleaned_data.questions)):
                prompt_key = f"prompt_{i}: {cleaned_data.questions[i]}"
                responses = cleaned_data.response_list[i]
                
                # Get list of log-prob lists: [[-0.1, -0.4], [-0.5, ...]]
                log_probs_list = cleaned_data.token_log_probs[i]

                #prompt-wise labels stored once per prompt
                p_false_i = None
                is_hall_i = None

                if hasattr(cleaned_data, "p_false"):
                    p_false_i = cleaned_data.p_false[i]  # Single value per prompt
                    p_false_i = float(p_false_i) if p_false_i is not None else None

                if hasattr(cleaned_data, "is_hallucination"):
                    is_hall_i = cleaned_data.is_hallucination[i]  # Single value per prompt
                    is_hall_i = bool(is_hall_i) if is_hall_i is not None else None
                

                # Group responses by semantic_ids (pre-existing clustering)
                prompt_clusters = group_responses_by_semantics(responses, cleaned_data.semantic_ids, log_probs_list)

                # --- NEW CODE ADDED HERE ---
                # Add p_false and is_hallucination as prompt-level metadata
                final_output[prompt_key] = {
                    "p_false": p_false_i,
                    "is_hallucination": is_hall_i,
                    "clusters": list(prompt_clusters.values())
                }
                # --- END OF NEW CODE ---

            # Save the results to a JSON file
            base_file = os.path.basename(file_path)
            output_name = base_file.replace('.pickle', f'_{metric}_clusters.json')
            output_filename = f"D:/LLM HALL/LLM-Hallucination/data/Authors/{output_name}"

            with open(output_filename, 'w') as f:
                json.dump(final_output, f, indent=2)

            print(f"Saved {metric} clusters to '{output_filename}'")
