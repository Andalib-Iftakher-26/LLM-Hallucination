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
├── experiments/
│   ├── analyze_results.py
│   ├── run_adaptive_budget.py
│   └── run_fixed_budget.py
├── pdfs/
├── src_code/
│   ├── bayesian_estimator/
│   │   ├── dirichlet.py
│   │   ├── estimator.py
│   │   ├── hierarchical_model.py
│   │   ├── truncated_dirichlet.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── compare_baseline.py
│   │   ├── metrics.py
│   │   └── visualize_results.py
│   ├── models/
│   │   ├── phi-2.Q4_K_M.gguf
│   │   ├── qwen1_5-0.5b-chat-q4_k_m.gguf
│   │   ├── tinyllama-1.1b-chat-v1.0.Q4_0.gguf
│   │   ├── phi.py
│   │   ├── qwen.py
│   │   └── tinyllama.py
│   ├── adaptive_sampler.py
│   ├── data_utils.py
│   ├── estimate_entropy.py
│   ├── meaning_mapper.py
│   └── train_prior.py
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






