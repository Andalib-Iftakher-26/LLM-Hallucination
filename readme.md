| 📄 / 📁                                                                                       | Description                                                                        |
| :-------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **📄 README.md**                                                                              | Main project documentation — explains setup, usage, and experiments.               |
| **📄 requirements.txt**                                                                       | Python dependencies required for the project.                                      |
| **📄 setup.py**                                                                               | Optional packaging file for `pip install -e .` (for local development).            |
|                                                                                               |                                                                                    |
| **📁 configs/**                                                                               | Centralized configuration files for models, datasets, estimators, and experiments. |
| ├── **base_config.yaml** — global defaults (device, logging, and sampling options).           |                                                                                    |
| ├── **model/** — defines LLM and entailment model settings (e.g., Llama-2, Mistral).          |                                                                                    |
| ├── **dataset/** — dataset-specific configs (TriviaQA, SQuAD, SVAMP, etc.).                   |                                                                                    |
| ├── **estimator/** — parameters for Bayesian, Histogram, and Rescaled estimators.             |                                                                                    |
| └── **experiments/** — pre-defined setups for fixed/adaptive budget experiments.              |                                                                                    |
|                                                                                               |                                                                                    |
| **📁 data/**                                                                                  | Stores all data artifacts (prompts, generations, meaning mappings, and results).   |
| ├── **prompts/** — input prompts for the LLM.                                                 |                                                                                    |
| ├── **generations/** — generated sequences and their probabilities *p(s‖x)*.                  |                                                                                    |
| ├── **meanings/** — semantic class mappings obtained via entailment models.                   |                                                                                    |
| └── **results/** — saved semantic entropy values, AUROC scores, and plots.                    |                                                                                    |
|                                                                                               |                                                                                    |
| **📁 src/**                                                                                   | Core implementation source code.                                                   |
| ├── **llm_interface.py** — handles LLM sampling and probability extraction.                   |                                                                                    |
| ├── **meaning_mapper.py** — clusters sequences into semantic meanings.                        |                                                                                    |
| ├── **data_utils.py** — helper functions for loading/saving datasets.                         |                                                                                    |
| ├── **train_prior.py** — learns prior distribution over meaning counts *K*.                   |                                                                                    |
| ├── **estimate_entropy.py** — main entry point for computing Bayesian semantic entropy.       |                                                                                    |
| ├── **adaptive_sampler.py** — dynamically allocates samples until target variance is reached. |                                                                                    |
| │                                                                                             |                                                                                    |
| ├── **bayesian_estimator/** — core Bayesian entropy estimation logic.                         |                                                                                    |
| │   ├── **dirichlet_entropy.py** — analytical Dirichlet expectation formulas.                 |                                                                                    |
| │   ├── **truncated_dirichlet.py** — Monte Carlo integration with constraints.                |                                                                                    |
| │   ├── **hierarchical_model.py** — Bayesian handling of unknown *K* values.                  |                                                                                    |
| │   ├── **estimator.py** — integrates all modules into a single estimator class.              |                                                                                    |
| │   └── **utils.py** — mathematical helpers for entropy and sampling.                         |                                                                                    |
| │                                                                                             |                                                                                    |
| └── **evaluation/** — scripts for assessing estimator performance.                            |                                                                                    |
|     ├── **metrics.py** — AUROC, F1, and statistical measures.                                 |                                                                                    |
|     ├── **compare_baselines.py** — compares Bayesian estimator vs. baselines.                 |                                                                                    |
|     └── **visualize_results.py** — creates plots for performance comparison.                  |                                                                                    |
|                                                                                               |                                                                                    |
| **📁 experiments/**                                                                           | Reproducible scripts and notebooks for replicating paper results.                  |
| ├── **run_fixed_budget.py** — runs fixed-sample (N) experiments.                              |                                                                                    |
| ├── **run_adaptive_budget.py** — runs adaptive-sampling experiments.                          |                                                                                    |
| └── **analyze_results.ipynb** — Jupyter notebook for analysis and visualization.              |                                                                                    |
|                                                                                               |                                                                                    |
| **📁 logs/**                                                                                  | Stores runtime and debugging logs.                                                 |
|                                                                                               |                                                                                    |
| **📁 tests/**                                                                                 | Unit and integration tests ensuring correctness and reproducibility.               |
| ├── **test_dirichlet_entropy.py**                                                             |                                                                                    |
| ├── **test_truncated_sampling.py**                                                            |                                                                                    |
| └── **test_end_to_end.py**                                                                    |                                                                                    |







##############################################################################################################################################


# 📂 Data Directory

This folder contains all datasets, model generations, semantic mappings, and entropy estimation results used in the **Bayesian Semantic Entropy** project.

Each subfolder represents a stage in the data pipeline — from prompts to final entropy outputs.

---

## 🧭 Directory Structure

```
data/
├── prompts/
│   ├── triviaqa_prompts.json
│   ├── squad_prompts.json
│   └── svamp_prompts.json
│
├── generations/
│   ├── triviaqa/
│   ├── squad/
│   └── svamp/
│
├── meanings/
│   ├── triviaqa/
│   ├── squad/
│   └── svamp/
│
└── results/
    ├── entropy_estimates.csv
    ├── bayesian_vs_baselines.csv
    └── figures/
```

---

## 🪶 1. `prompts/` — Input Questions or Contexts

Contains the **raw prompts** that the model will answer.

Example (`triviaqa_prompts.json`):

```json
[
  { "id": "tqa_001", "question": "What is the capital of France?" },
  { "id": "tqa_002", "question": "Who wrote the play Hamlet?" }
]
```

Used by: `src/llm_interface.py`

---

## 🤖 2. `generations/` — Model Responses

Stores **LLM-generated responses** for each prompt, along with optional log probabilities.

Example (`generations/triviaqa/sample_001.json`):

```json
{
  "id": "tqa_001",
  "prompt": "What is the capital of France?",
  "responses": [
    {"text": "Paris.", "logprob": -0.3},
    {"text": "The capital of France is Paris.", "logprob": -0.5}
  ]
}
```

Generated by: `src/llm_interface.py`

---

## 🧠 3. `meanings/` — Semantic Clusters

After generation, responses are grouped by **semantic equivalence** using an entailment model.

Example (`meanings/triviaqa/meanings_001.json`):

```json
{
  "id": "tqa_001",
  "prompt": "What is the capital of France?",
  "clusters": [
    {
      "meaning_id": 0,
      "members": [
        "Paris.",
        "The capital of France is Paris.",
        "It's Paris."
      ]
    }
  ]
}
```

Generated by: `src/meaning_mapper.py`

---

## 📊 4. `results/` — Entropy & Evaluation Outputs

Contains computed **semantic entropy estimates**, baseline comparisons, and figures.

Example (`entropy_estimates.csv`):

| prompt_id | dataset  | estimator | E[h] | Var[h] | K_estimated | N_samples |
| --------- | -------- | --------- | ---- | ------ | ----------- | --------- |
| tqa_001   | triviaqa | bayesian  | 0.21 | 0.002  | 1           | 10        |
| tqa_002   | triviaqa | bayesian  | 0.48 | 0.005  | 3           | 12        |

Plots and comparison charts are saved in `results/figures/`.

Generated by:

* `src/estimate_entropy.py`
* `src/evaluation/compare_baselines.py`
* `src/evaluation/visualize_results.py`

---

## 🔄 Data Flow Summary

| Step                     | Script                 | Input                       | Output                 |
| ------------------------ | ---------------------- | --------------------------- | ---------------------- |
| 1️⃣ Generate responses   | `llm_interface.py`     | `prompts/`                  | `generations/`         |
| 2️⃣ Map meanings         | `meaning_mapper.py`    | `generations/`              | `meanings/`            |
| 3️⃣ Train prior          | `train_prior.py`       | `meanings/`                 | `configs/prior_K.json` |
| 4️⃣ Estimate entropy     | `estimate_entropy.py`  | `meanings/`, `prior_K.json` | `results/`             |
| 5️⃣ Visualize / evaluate | `compare_baselines.py` | `results/`                  | `results/figures/`     |

---

## 🧩 Notes

* Keep raw data (prompts and generations) under version control only if small.
  Large datasets should be added to `.gitignore`.
* Use consistent prompt IDs across all files (`id` field must match).
* Store probabilities (`logprob`) whenever possible — they improve Bayesian truncation accuracy.

---

**Next step:** run `src/llm_interface.py` to populate the `generations/` folder with model outputs.




##############################################################################################################################################
