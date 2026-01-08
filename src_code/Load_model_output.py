# found LLM generated data by the original authors updating pipeline as such

import pickle
import random
import numpy as np
from collections import Counter
from collections import namedtuple

# paths

path1 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-2-7b-chat--nq--results.pickle'
path2 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-2-7b-chat--squad--results.pickle'
path3 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-2-7b-chat--svamp--results.pickle'
path4 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-2-7b-chat--trivia_qa--results.pickle'

path5 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-3.2-3B-Instruct--nq--results.pickle'
path6 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-3.2-3B-Instruct--squad--results.pickle'
path7 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-3.2-3B-Instruct--svamp--results.pickle'
path8 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-3.2-3B-Instruct--trivia_qa--results.pickle'
path9 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Llama-3.3-70B-Instruct--trivia_qa--results.pickle'

path10 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Mistral-Small-24B-Instruct-2501--nq--results.pickle'
path11 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Mistral-Small-24B-Instruct-2501--squad--results.pickle'
path12 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Mistral-Small-24B-Instruct-2501--svamp--results.pickle'
path13 = 'D:/LLM HALL/LLM-Hallucination/data/generations/nnnp-Mistral-Small-24B-Instruct-2501--trivia_qa--results.pickle'

# global vars 
no_of_train_prompts = 200
truncate_no_semantic_classes = 7
no_eval_seeds = 5
num_mc_samples = 1000





# think of them as dataset features
Dataset = namedtuple("Dataset", ["is_hallucination", "semantic_ids", "response_list", 
                                 "token_log_probs", "p_false", "most_likely_answer", 
                                 "questions", "contexts"]) 


#for experiment reproducibility
random.seed(42)
np.random.seed(42)

# what i actually need for my pipeline
CleanedDataset = namedtuple("CleanedDataset", 
                            ["is_hallucination", 
                            "response_list", 
                            "token_log_probs", 
                            "questions", 
                            "contexts",
                            'p_false',
                        ]
                            )





def create_cleaned_dataset(path: str) -> CleanedDataset:
    # loading data    
    with open(path, 'rb') as file:
        data = pickle.load(file)

    # creating a new dataset object with only the fields we need    
    clean_data = CleanedDataset(
    is_hallucination = data[0],
    response_list    = data[2],
    token_log_probs  = data[3],
    p_false          = data[4],
    questions        = data[6],
    contexts         = data[7]
    )
    return clean_data




# what i actually need for my pipeline
DatawithSemantic = namedtuple("CleanedDataset", 
                            ["is_hallucination", 
                            "response_list", 
                            "token_log_probs", 
                            "questions", 
                            "contexts",
                            'p_false',
                            "semantic_ids",]
                            )

def originalData(path: str) -> DatawithSemantic:
    # loading data    
    with open(path, 'rb') as file:
        data = pickle.load(file)

    # creating a new dataset object with only the fields we need    
    original_data = DatawithSemantic(
    is_hallucination = data[0],
    semantic_ids     = data[1],
    response_list    = data[2],
    token_log_probs  = data[3],
    p_false          = data[4],
    questions        = data[6],
    contexts         = data[7],
    
    )
    return original_data


