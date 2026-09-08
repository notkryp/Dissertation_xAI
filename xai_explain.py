"""
Chapter 4 (continued) - Explainability (SHAP + LIME) Implementation
Explainable AI for Software Defect Prediction

WHY EACH PARAMETER IS SET THE WAY IT IS (for viva prep - keep this reasoning):

- TreeSHAP (Random Forest, XGBoost): uses feature_perturbation="tree_path_dependent",
  which computes EXACT Shapley values directly from tree structure, needs NO
  background dataset, and avoids a known SHAP/XGBoost incompatibility where the
  alternative "interventional" method does not yet support XGBoost's categorical
  split handling. No sampling/capping applied - full test set is explained for
  both tree models.

  IMPORTANT: tree_path_dependent's additive output space differs by model type:
    - Random Forest (sklearn): leaves store probability estimates directly, so
      the natively comparable output is model.predict_proba()[:, 1].
    - XGBoost: leaves store additive log-odds ("margin") contributions, so the
      natively comparable output is model.predict(X, output_margin=True).
  get_native_raw_output() below handles this distinction so the fidelity check
  compares SHAP's reconstruction against the CORRECT output space per model.

- Kernel SHAP (SVM, Neural Network/MLPClassifier): non-tree models, so SHAP must
  estimate Shapley values via perturbation sampling, which is combinatorially
  expensive by construction (true on any hardware). Three standard,
  literature-backed knobs keep it tractable without weakening validity:
    1. BACKGROUND_SIZE = 100   -> background summarised via k-means (shap.kmeans),
       SHAP's own documented recommendation (Lundberg & Lee, 2017).
    2. N_LOCAL_INSTANCES = 100 -> capped at dataset test size where smaller.
       Within the commonly used 50-200 range in published XAI-for-SDP studies.
    3. NSAMPLES = 200          -> reduced from SHAP's "auto" default (~2076) to
       keep computation tractable given JM1's SVM being ~400x slower per-
       prediction than the other datasets' SVMs (Chapter 4, Section 4.7).

- LIME: applied uniformly to ALL FOUR models (model-agnostic by design), same
  N_LOCAL_INSTANCES=100 sample for consistency. num_samples=1000 per explanation
  (LIME's own common default).

- STABILITY computed on a smaller sub-sample (STABILITY_INSTANCES=20) by
  re-running the SAME explanation twice with different random seeds and
  measuring Spearman rank correlation between the two feature rankings.

Before running:
    pip install shap lime joblib pandas numpy scikit-learn xgboost scipy

Outputs:
    results/global_importance_<dataset>_<model>_shap.csv
    results/explanation_quality.csv
    results/local_explanations_sample.csv
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import shap
from lime.lime_tabular import LimeTabularExplainer

DATASETS = ["cm1", "kc1", "jm1", "pc1"]
TREE_MODELS = ["RandomForest", "GradientBoosting_XGBoost"]
KERNEL_MODELS = ["SVM", "NeuralNetwork"]
ALL_MODELS = TREE_MODELS + KERNEL_MODELS

RANDOM_SEED = 42
BACKGROUND_SIZE = 100
N_LOCAL_INSTANCES = 100
NSAMPLES = 200
STABILITY_INSTANCES = 20
LIME_NUM_SAMPLES = 1000

os.makedirs("results", exist_ok=True)


def load_model(dataset, model_key):
    return joblib.load(f"models/{dataset}_{model_key}.pkl")


def load_data(dataset, model_key):
    suffix = "scaled" if model_key in KERNEL_MODELS else ""
    train_file = f"data/processed/{dataset}_train_smote" + ("_scaled" if suffix else "") + ".csv"
    test_file = f"data/processed/{dataset}_test" + ("_scaled" if suffix else "") + ".csv"
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    X_train, y_train = train.drop(columns=["defects"]), train["defects"]
    X_test, y_test = test.drop(columns=["defects"]), test["defects"]
    return X_train, y_train, X_test, y_test


def predict_proba_positive(model, X):
    return model.predict_proba(X)[:, 1]


def normalize_shap_output(shap_values, expected_value, positive_class_idx=1):
    """Handle both older SHAP (list of per-class arrays) and newer SHAP
    (single 3D ndarray: samples x features x classes) output formats."""
    if isinstance(shap_values, list):
        sv = shap_values[positive_class_idx]
        ev = expected_value[positive_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, positive_class_idx]
        ev_arr = np.array(expected_value)
        ev = ev_arr[positive_class_idx] if ev_arr.size > 1 else float(ev_arr)
    else:
        sv = shap_values
        ev_arr = np.array(expected_value) if isinstance(expected_value, (list, np.ndarray)) else None
        ev = ev_arr[positive_class_idx] if ev_arr is not None and ev_arr.size > 1 else expected_value
    return sv, ev


def get_native_raw_output(model, X, model_key):
    """Returns the output space that matches TreeSHAP's tree_path_dependent
    additive reconstruction for each model type (see module docstring)."""
    if model_key == "RandomForest":
        return model.predict_proba(X)[:, 1]
    elif model_key == "GradientBoosting_XGBoost":
        return model.predict(X, output_margin=True)
    else:
        return predict_proba_positive(model, X)


def run_tree_shap(model, X_test, dataset, model_key, results):
    """Exact SHAP for tree ensembles via tree_path_dependent (no background
    needed, avoids SHAP/XGBoost categorical-split incompatibility). Full test
    set explained - no sampling/capping."""
    print(f"  [SHAP-Tree] {dataset}/{model_key}: explaining full test set ({len(X_test)} instances)")
    t0 = time.time()
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    raw_shap_values = explainer.shap_values(X_test)
    shap_values, expected_value = normalize_shap_output(raw_shap_values, explainer.expected_value)
    elapsed = time.time() - t0

    reconstructed = shap_values.sum(axis=1) + expected_value
    actual = get_native_raw_output(model, X_test, model_key)
    fidelity_mae = np.mean(np.abs(reconstructed - actual))

    global_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    global_importance.insert(0, "model", model_key)
    global_importance.insert(0, "dataset", dataset)
    global_importance.to_csv(f"results/global_importance_{dataset}_{model_key}_shap.csv", index=False)

    results.append({
        "dataset": dataset, "model": model_key, "method": "SHAP",
        "fidelity_mae": fidelity_mae, "fidelity_r2_mean": None, "cost_seconds": elapsed,
        "n_instances_explained": len(X_test), "stability_spearman": None
    })
    return shap_values


def run_kernel_shap(model, X_train, X_test, dataset, model_key, results):
    n_local = min(N_LOCAL_INSTANCES, len(X_test))
    X_sample = X_test.sample(n=n_local, random_state=RANDOM_SEED)
    print(f"  [SHAP-Kernel] {dataset}/{model_key}: background={BACKGROUND_SIZE}, "
          f"instances={n_local}, nsamples={NSAMPLES}")

    background = shap.kmeans(X_train, min(BACKGROUND_SIZE, len(X_train)))

    t0 = time.time()
    explainer = shap.KernelExplainer(lambda x: predict_proba_positive(model, x), background)
    shap_values = explainer.shap_values(X_sample, nsamples=NSAMPLES)
    elapsed = time.time() - t0

    reconstructed = shap_values.sum(axis=1) + explainer.expected_value
    actual = predict_proba_positive(model, X_sample)
    fidelity_mae = np.mean(np.abs(reconstructed - actual))

    n_stab = min(STABILITY_INSTANCES, n_local)
    X_stab = X_sample.iloc[:n_stab]
    explainer2 = shap.KernelExplainer(lambda x: predict_proba_positive(model, x), background)
    shap_values_rerun = explainer2.shap_values(X_stab, nsamples=NSAMPLES)
    correlations = []
    for i in range(n_stab):
        rho, _ = spearmanr(np.abs(shap_values[i]), np.abs(shap_values_rerun[i]))
        if not np.isnan(rho):
            correlations.append(rho)
    stability = np.mean(correlations) if correlations else None

    global_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    global_importance.insert(0, "model", model_key)
    global_importance.insert(0, "dataset", dataset)
    global_importance.to_csv(f"results/global_importance_{dataset}_{model_key}_shap.csv", index=False)

    results.append({
        "dataset": dataset, "model": model_key, "method": "SHAP",
        "fidelity_mae": fidelity_mae, "fidelity_r2_mean": None, "cost_seconds": elapsed,
        "n_instances_explained": n_local, "stability_spearman": stability
    })
    return shap_values, X_sample


def run_lime(model, X_train, X_test, dataset, model_key, results):
    n_local = min(N_LOCAL_INSTANCES, len(X_test))
    X_sample = X_test.sample(n=n_local, random_state=RANDOM_SEED)
    print(f"  [LIME] {dataset}/{model_key}: instances={n_local}, num_samples={LIME_NUM_SAMPLES}")

    explainer = LimeTabularExplainer(
        X_train.values, feature_names=list(X_train.columns),
        class_names=["clean", "defective"], mode="classification",
        random_state=RANDOM_SEED
    )

    fidelities, local_top_features = [], []
    t0 = time.time()
    for idx in range(n_local):
        instance = X_sample.iloc[idx].values
        exp = explainer.explain_instance(
            instance, model.predict_proba, num_features=len(X_train.columns),
            num_samples=LIME_NUM_SAMPLES
        )
        fidelities.append(exp.score)
        top3 = sorted(exp.as_list(), key=lambda t: abs(t[1]), reverse=True)[:3]
        local_top_features.append({"dataset": dataset, "model": model_key,
                                    "instance_index": idx, "top_features": top3})
    elapsed = time.time() - t0

    n_stab = min(STABILITY_INSTANCES, n_local)
    explainer2 = LimeTabularExplainer(
        X_train.values, feature_names=list(X_train.columns),
        class_names=["clean", "defective"], mode="classification",
        random_state=RANDOM_SEED + 1
    )
    correlations = []
    for idx in range(n_stab):
        instance = X_sample.iloc[idx].values
        exp1 = explainer.explain_instance(instance, model.predict_proba,
                                           num_features=len(X_train.columns), num_samples=LIME_NUM_SAMPLES)
        exp2 = explainer2.explain_instance(instance, model.predict_proba,
                                            num_features=len(X_train.columns), num_samples=LIME_NUM_SAMPLES)
        d1 = dict(exp1.as_list())
        d2 = dict(exp2.as_list())
        common_feats = [f for f in d1 if f in d2]
        if len(common_feats) > 1:
            v1 = [abs(d1[f]) for f in common_feats]
            v2 = [abs(d2[f]) for f in common_feats]
            rho, _ = spearmanr(v1, v2)
            if not np.isnan(rho):
                correlations.append(rho)
    stability = np.mean(correlations) if correlations else None

    results.append({
        "dataset": dataset, "model": model_key, "method": "LIME",
        "fidelity_mae": None, "fidelity_r2_mean": np.mean(fidelities),
        "cost_seconds": elapsed, "n_instances_explained": n_local,
        "stability_spearman": stability
    })
    return pd.DataFrame(local_top_features)


def main():
    quality_results = []
    all_local_lime = []

    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} ===")
        for model_key in ALL_MODELS:
            model = load_model(dataset, model_key)
            X_train, y_train, X_test, y_test = load_data(dataset, model_key)

            if model_key in TREE_MODELS:
                run_tree_shap(model, X_test, dataset, model_key, quality_results)
            else:
                run_kernel_shap(model, X_train, X_test, dataset, model_key, quality_results)

            lime_df = run_lime(model, X_train, X_test, dataset, model_key, quality_results)
            all_local_lime.append(lime_df)

    quality_df = pd.DataFrame(quality_results)
    quality_df.to_csv("results/explanation_quality.csv", index=False)

    local_df = pd.concat(all_local_lime, ignore_index=True)
    local_df.to_csv("results/local_explanations_sample.csv", index=False)

    print("\nAll done.")
    print(quality_df)


if __name__ == "__main__":
    main()