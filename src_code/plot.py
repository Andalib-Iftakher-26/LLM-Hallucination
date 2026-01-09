
def save_entropy_diagnostics(input_dir: str, output_dir: str):
    import os, json, pickle, math
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Optional sklearn for ROC/PR/Calibration
    try:
        from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
        from sklearn.calibration import calibration_curve
        SKLEARN_OK = True
    except Exception:
        SKLEARN_OK = False

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def _load_results(prefix: str):
        """Try json then pickle for given prefix inside input_dir."""
        candidates = [
            os.path.join(input_dir, f"{prefix}_results.json"),
            os.path.join(input_dir, f"{prefix}_results.pkl"),
            os.path.join(input_dir, f"{prefix}_results.pickle"),
        ]
        for fp in candidates:
            if os.path.exists(fp):
                if fp.endswith(".json"):
                    with open(fp, "r", encoding="utf-8") as f:
                        return json.load(f), fp
                else:
                    with open(fp, "rb") as f:
                        return pickle.load(f), fp
        return None, None

    def _to_df(results_obj, run_name: str):
        """Normalize dict/list results into a DataFrame."""
        rows = []
        if results_obj is None:
            return pd.DataFrame()

        if isinstance(results_obj, dict):
            # common case: {prompt_id: {...}}
            for prompt_id, d in results_obj.items():
                if not isinstance(d, dict):
                    continue
                rows.append({
                    "run": run_name,
                    "prompt_id": str(prompt_id),
                    "entropy": d.get("entropy", np.nan),
                    "variance": d.get("variance", np.nan),
                    "samples_used": d.get("samples_used", np.nan),
                    "p_false": d.get("p_false", np.nan),
                    "is_hallucination": d.get("is_hallucination", np.nan),
                })
        elif isinstance(results_obj, list):
            # list of dict records
            for i, d in enumerate(results_obj):
                if not isinstance(d, dict):
                    continue
                rows.append({
                    "run": run_name,
                    "prompt_id": str(d.get("prompt_id", f"row_{i}")),
                    "entropy": d.get("entropy", np.nan),
                    "variance": d.get("variance", np.nan),
                    "samples_used": d.get("samples_used", np.nan),
                    "p_false": d.get("p_false", np.nan),
                    "is_hallucination": d.get("is_hallucination", np.nan),
                })

        df = pd.DataFrame(rows)

        # Coerce numeric columns
        for col in ["entropy", "variance", "samples_used", "p_false"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalize label to {0,1} where possible
        def _lab(x):
            if isinstance(x, (bool, np.bool_)):
                return int(x)
            if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
                return int(x != 0)
            if isinstance(x, str):
                xl = x.strip().lower()
                if xl in ("true", "t", "yes", "y", "1"):
                    return 1
                if xl in ("false", "f", "no", "n", "0"):
                    return 0
            return np.nan

        df["label"] = df["is_hallucination"].map(_lab) if "is_hallucination" in df.columns else np.nan
        return df

    def _savefig(filename: str):
        fp = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(fp, dpi=200)
        plt.close()
        return fp

    # ---- Load baseline/candidate if present
    base_obj, base_fp = _load_results("baseline")
    cand_obj, cand_fp = _load_results("candidate")

    if base_obj is None and cand_obj is None:
        raise FileNotFoundError(
            f"No results files found in input_dir={input_dir}\n"
            "Expected baseline_results.(json|pkl|pickle) and/or candidate_results.(json|pkl|pickle)."
        )

    df_base = _to_df(base_obj, "baseline") if base_obj is not None else pd.DataFrame()
    df_cand = _to_df(cand_obj, "candidate") if cand_obj is not None else pd.DataFrame()
    df = pd.concat([df_base, df_cand], ignore_index=True) if (len(df_base) or len(df_cand)) else pd.DataFrame()

    # ---- Basic filtering
    df = df.copy()
    df = df[np.isfinite(df["entropy"])]

    metrics = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "files_loaded": {"baseline": base_fp, "candidate": cand_fp},
        "n_rows": int(len(df)),
        "n_baseline": int(len(df_base)),
        "n_candidate": int(len(df_cand)),
        "sklearn_available": SKLEARN_OK,
        "plots": [],
    }

    # ---------------- Plot 1: Entropy histograms (by label if available) ----------------
    for run_name, dfr in df.groupby("run"):
        if dfr.empty:
            continue
        has_label = dfr["label"].notna().any()

        plt.figure(figsize=(8, 4))
        if has_label:
            for lab, color, name in [(0, "seagreen", "Not hallucination"), (1, "crimson", "Hallucination")]:
                subset = dfr.loc[dfr["label"] == lab, "entropy"].dropna()
                if len(subset):
                    plt.hist(subset, bins=25, alpha=0.45, color=color, label=name)
            plt.legend()
            plt.title(f"Entropy distribution by label ({run_name})")
        else:
            plt.hist(dfr["entropy"].dropna(), bins=25, alpha=0.7, color="steelblue")
            plt.title(f"Entropy distribution ({run_name})")

        plt.xlabel("Entropy")
        plt.ylabel("Count")
        metrics["plots"].append(_savefig(f"entropy_hist_{run_name}.png"))

    # ---------------- Plot 2: Entropy vs p_false ----------------
    for run_name, dfr in df.groupby("run"):
        if dfr.empty or dfr["p_false"].dropna().empty:
            continue
        plt.figure(figsize=(6.5, 5))
        colors = dfr["label"].map({0: "seagreen", 1: "crimson"}).fillna("gray")
        plt.scatter(dfr["entropy"], dfr["p_false"], c=colors, alpha=0.75, edgecolor="none")
        plt.xlabel("Entropy")
        plt.ylabel("p_false")
        plt.title(f"Entropy vs p_false ({run_name})")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        metrics["plots"].append(_savefig(f"entropy_vs_pfalse_{run_name}.png"))

    # ---------------- Plot 3: Entropy vs samples_used ----------------
    for run_name, dfr in df.groupby("run"):
        if dfr.empty or dfr["samples_used"].dropna().empty:
            continue
        plt.figure(figsize=(6.5, 5))
        colors = dfr["label"].map({0: "seagreen", 1: "crimson"}).fillna("gray")
        plt.scatter(dfr["samples_used"], dfr["entropy"], c=colors, alpha=0.75, edgecolor="none")
        plt.xlabel("samples_used")
        plt.ylabel("Entropy")
        plt.title(f"Entropy vs samples_used ({run_name})")
        plt.grid(alpha=0.3)
        metrics["plots"].append(_savefig(f"entropy_vs_samples_{run_name}.png"))

    # ---------------- Plot 4: Variance vs samples_used ----------------
    for run_name, dfr in df.groupby("run"):
        if dfr.empty or dfr["samples_used"].dropna().empty or dfr["variance"].dropna().empty:
            continue
        plt.figure(figsize=(6.5, 5))
        colors = dfr["label"].map({0: "seagreen", 1: "crimson"}).fillna("gray")
        plt.scatter(dfr["samples_used"], dfr["variance"], c=colors, alpha=0.75, edgecolor="none")
        plt.xlabel("samples_used")
        plt.ylabel("Variance")
        plt.title(f"Variance vs samples_used ({run_name})")
        plt.grid(alpha=0.3)
        metrics["plots"].append(_savefig(f"variance_vs_samples_{run_name}.png"))

    # ---------------- Plot 5: ROC + PR for entropy (if labels + sklearn) ----------------
    if SKLEARN_OK:
        for run_name, dfr in df.groupby("run"):
            dfr = dfr.dropna(subset=["entropy", "label"])
            if dfr.empty or dfr["label"].nunique() < 2:
                continue

            y = dfr["label"].astype(int).values
            score = dfr["entropy"].values

            # ROC
            try:
                auc = float(roc_auc_score(y, score))
                fpr, tpr, _ = roc_curve(y, score)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
                plt.plot([0, 1], [0, 1], "--", alpha=0.5)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC (Entropy) — {run_name}")
                plt.legend()
                plt.grid(alpha=0.3)
                metrics["plots"].append(_savefig(f"roc_entropy_{run_name}.png"))
                metrics.setdefault("auroc_entropy", {})[run_name] = auc
            except Exception:
                pass

            # PR
            try:
                ap = float(average_precision_score(y, score))
                prec, rec, _ = precision_recall_curve(y, score)
                plt.figure(figsize=(6, 6))
                plt.plot(rec, prec, label=f"Avg Precision = {ap:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"PR Curve (Entropy) — {run_name}")
                plt.legend()
                plt.grid(alpha=0.3)
                metrics["plots"].append(_savefig(f"pr_entropy_{run_name}.png"))
                metrics.setdefault("ap_entropy", {})[run_name] = ap
            except Exception:
                pass

        # Calibration for p_false (if present)
        for run_name, dfr in df.groupby("run"):
            dfr = dfr.dropna(subset=["p_false", "label"])
            if dfr.empty or dfr["label"].nunique() < 2:
                continue

            y = dfr["label"].astype(int).values
            p = np.clip(dfr["p_false"].astype(float).values, 0.0, 1.0)

            try:
                frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
                plt.figure(figsize=(6, 6))
                plt.plot(mean_pred, frac_pos, marker="o")
                plt.plot([0, 1], [0, 1], "--", alpha=0.5)
                plt.xlabel("Mean predicted p_false")
                plt.ylabel("Empirical hallucination rate")
                plt.title(f"Calibration: p_false — {run_name}")
                plt.grid(alpha=0.3)
                metrics["plots"].append(_savefig(f"calibration_pfalse_{run_name}.png"))
            except Exception:
                pass

    # ---------------- Plot 6: Compare baseline vs candidate (if both present) ----------------
    if not df_base.empty and not df_cand.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(df_base["entropy"].dropna(), bins=30, alpha=0.45, label="baseline", color="slateblue")
        plt.hist(df_cand["entropy"].dropna(), bins=30, alpha=0.45, label="candidate", color="darkorange")
        plt.xlabel("Entropy")
        plt.ylabel("Count")
        plt.title("Entropy distribution comparison (baseline vs candidate)")
        plt.legend()
        plt.grid(alpha=0.25)
        metrics["plots"].append(_savefig("entropy_hist_compare.png"))

        if df_base["p_false"].dropna().any() and df_cand["p_false"].dropna().any():
            plt.figure(figsize=(6.5, 5))
            plt.scatter(df_base["entropy"], df_base["p_false"], alpha=0.55, label="baseline", color="slateblue")
            plt.scatter(df_cand["entropy"], df_cand["p_false"], alpha=0.55, label="candidate", color="darkorange")
            plt.xlabel("Entropy")
            plt.ylabel("p_false")
            plt.title("Entropy vs p_false (baseline vs candidate)")
            plt.ylim(0, 1)
            plt.grid(alpha=0.3)
            plt.legend()
            metrics["plots"].append(_savefig("scatter_entropy_compare.png"))

    # ---- Save metrics summary
    metrics_fp = os.path.join(output_dir, "metrics.json")
    with open(metrics_fp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics





metrics = save_entropy_diagnostics(
    input_dir="/path/to/results",
    output_dir="/path/to/save/plots"
)
print("Saved plots:", metrics["plots"])
