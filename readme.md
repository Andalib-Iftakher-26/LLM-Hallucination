# BAYESIAN ENTROPY FOR LLM HALLUCINATION DETECTION

# 🧠 LLM-HALLUCINATION

This repository explores **hallucination detection and mitigation** in Large Language Models (LLMs) using Bayesian estimators, adaptive sampling, and evaluation over multiple QA datasets (SQuAD, SVAMP, TriviaQA).

---

## 📁 Project Structure

LLM-HALLUCINATION/
├── .vscode/ # VS Code workspace settings
│ └── settings.json
│
├── configs/ # Configuration files for datasets, estimators, and experiments
│ ├── dataset/
│ │ └── triviaqa.yaml
│ ├── estimator/
│ │ ├── bayes_default.yaml
│ │ ├── histogram.yaml
│ │ └── rescaled.yaml
│ ├── experiments/
│ │ ├── adaptive_budget.yaml
│ │ └── fixed_budget.yaml
│ └── model/
│ ├── base_config.yaml
│ ├── llama2.yaml
│ └── mistral.yaml
│
├── data/ # Input data, prompt templates, and evaluation outputs
│ ├── generations/
│ ├── meanings/
│ ├── prompts/
│ │ ├── squad_prompts.json
│ │ ├── svamp_prompts.json
│ │ └── triviaqa_prompts.json
│ └── results/
│ ├── SQuAD/
│ │ ├── dev-v2.0.json
│ │ └── SQuAD.ipynb
│ ├── SVAMP/
│ │ ├── SVAMP.json
│ │ └── SVAMP.ipynb
│ └── TriviaQA/
│ ├── TriviaQA.json
│ └── TriviaQA.ipynb
│
├── experiments/ # Scripts for running and analyzing experiments
│ ├── analyze_results.py
│ ├── run_adaptive_budget.py
│ └── run_fixed_budget.py
│
├── pdfs/ # Generated reports and visualizations (optional)
│
├── src_code/ # Core source code
│ ├── bayesian_estimator/ # Bayesian modeling and inference modules
│ │ ├── dirichlet.py
│ │ ├── estimator.py
│ │ ├── hierarchical_model.py
│ │ ├── truncated_dirichlet.py
│ │ └── utils.py
│ │
│ ├── evaluation/ # Evaluation and metric computation
│ │ ├── compare_baseline.py
│ │ ├── metrics.py
│ │ └── visualize_results.py
│ │
│ ├── models/ # Model interfaces and weights
│ │ ├── phi-2.Q4_K_M.gguf
│ │ ├── qwen1_5-0.5b-chat-q4_k_m.gguf
│ │ ├── tinyllama-1.1b-chat-v1.0.Q4_0.gguf
│ │ ├── phi.py
│ │ ├── qwen.py
│ │ └── tinyllama.py
│ │
│ ├── adaptive_sampler.py # Adaptive sampling strategies for hallucination detection
│ ├── data_utils.py # Data preprocessing and I/O utilities
│ ├── estimate_entropy.py # Entropy-based uncertainty estimation
│ ├── meaning_mapper.py # Semantic similarity mapping between generations and references
│ └── train_prior.py # Prior model training for Bayesian estimator
│
├── .gitignore # Ignore unnecessary files
├── .gitattributes
├── requirements.txt # Python dependencies
├── README.md # (You are here)
└── setup_instructions.md # Optional detailed setup guide






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






