# BAYESIAN ENTROPY FOR LLM HALLUCINATION DETECTION

# 🧠 LLM-HALLUCINATION

This repository explores **hallucination detection and mitigation** in Large Language Models (LLMs) using Bayesian estimators, adaptive sampling, and evaluation over multiple QA datasets (SQuAD, SVAMP, TriviaQA).

---

## 📁 Project Structure
```
LLM-HALLUCINATION/
├── .vscode/
│   └── settings.json
|
├── data/
│   ├── generations/
│   ├── meanings/
|   |   ├── cosine_sim
|   |   ├── pearson_sim
|   |   ├── rbf_sim
│   ├── prompts/
│   │   ├── squad_prompts.json
│   │   ├── svamp_prompts.json
│   │   └── triviaqa_prompts.json
│   └── results/
│       ├── SQuAD/
│       │   ├── dev-v2.0.json
│       │   └── SQuAD.ipynb
│       ├── SVAMP/
│       │   ├── SVAMP.json
│       │   └── SVAMP.ipynb
│       └── TriviaQA/
│           ├── TriviaQA.json
│           └── TriviaQA.ipynb
├── pdfs/
├── src_code/
│   ├── bayesian_estimator.py
|   ├── eval_tune.py
|   ├── load_model_output.py
|   ├── meaning_mapper.py
|   ├── original_clusters.py 
|   ├── run_adaptive.py 
|      
├── .gitignore
├── .gitattributes
├── requirements.txt
├── README.md

```




---

## ⚙️ Setup

### 1. Create Conda Environment
```bash
conda create -n llmhall python=3.11
conda activate llmhall
pip install -r requirements.txt
```

If you’re using llama-cpp-python or similar:  Install Visual Studio Build Tools with Desktop Development with C++.
```pip install llama-cpp-python --force-reinstall --no-cache-dir```

---



## THEORY
---
 ## step 1

For each prompt, generate multiple responses from the language model

Group the responses into semantic clusters (each cluster represents one meaning)

Let the number of observed meanings be k_obs


 ## step 2

For each prompt, construct a probability distribution over the possible total number of meanings K, conditioned on k_obs

Enforce the constraint K ≥ k_min, where k_min is the minimum number of meanings observed for the prompt, k_min = k_obs


 ## step 3

For each possible value of K, construct a Dirichlet distribution over the K meaning probabilities

Enforce lower bounds on meaning probabilities using the summed likelihoods of observed sequences belonging to each meaning


 ## step 4

For each Dirichlet distribution, compute Shannon entropy

This induces a probability distribution over entropy values

Integrate hierarchically over K to compute the expected semantic entropy and the variance of semantic entropy



 ## step 5

Use the estimated semantic entropy as a signal to determine whether a response is likely reliable or hallucinated

---






