"""
Chapter 4 - Model Training Script
Explainable AI for Software Defect Prediction

Trains Random Forest, XGBoost (Gradient Boosting), SVM, and Neural Network
(scikit-learn MLPClassifier) on the four pre-processed datasets (CM1, KC1,
JM1, PC1), with grid search hyperparameter tuning, and evaluates on the
held-out test sets.

Before running:
    pip install pandas numpy scikit-learn xgboost joblib

Folder structure expected (from preprocess.py output):
    data/processed/<name>_train_smote.csv
    data/processed/<name>_test.csv
    data/processed/<name>_train_smote_scaled.csv
    data/processed/<name>_test_scaled.csv

Outputs:
    results/model_results.csv       - one row per (dataset, model) with best
                                       hyperparameters and test-set metrics
    models/<dataset>_<model>.pkl    - saved trained models (for Chapter 4 XAI script)
"""

import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, matthews_corrcoef)
from xgboost import XGBClassifier

DATASETS = ["cm1", "kc1", "jm1", "pc1"]
RANDOM_SEED = 42
CV_FOLDS = 5

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)


def load_split(name, suffix):
    df = pd.read_csv(f"data/processed/{name}_{suffix}.csv")
    y = df["defects"]
    X = df.drop(columns=["defects"])
    return X, y


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def run_grid(name, model_key, estimator, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(estimator, param_grid, cv=CV_FOLDS, scoring="f1", n_jobs=-1)
    t0 = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - t0

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    joblib.dump(best, f"models/{name}_{model_key}.pkl")

    metrics = evaluate(y_test, y_pred, y_prob)
    metrics.update({"dataset": name, "model": model_key,
                     "best_params": grid.best_params_, "train_time_s": round(train_time, 2)})
    return metrics


def main():
    all_results = []

    for name in DATASETS:
        print(f"\n=== {name.upper()} ===")

        X_train, y_train = load_split(name, "train_smote")
        X_test, y_test = load_split(name, "test")

        print("Training Random Forest...")
        rf_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}
        all_results.append(run_grid(name, "RandomForest",
                                     RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
                                     rf_grid, X_train, y_train, X_test, y_test))

        print("Training XGBoost...")
        xgb_grid = {"learning_rate": [0.01, 0.1, 0.3],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7]}
        all_results.append(run_grid(name, "GradientBoosting_XGBoost",
                                     XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss"),
                                     xgb_grid, X_train, y_train, X_test, y_test))

        X_train_s, y_train_s = load_split(name, "train_smote_scaled")
        X_test_s, y_test_s = load_split(name, "test_scaled")

        print("Training SVM...")
        svm_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}
        all_results.append(run_grid(name, "SVM",
                                     SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
                                     svm_grid, X_train_s, y_train_s, X_test_s, y_test_s))

        print("Training Neural Network (MLPClassifier)...")
        nn_grid = {"hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
                   "learning_rate_init": [0.001, 0.01]}
        all_results.append(run_grid(name, "NeuralNetwork",
                                     MLPClassifier(max_iter=500, random_state=RANDOM_SEED, early_stopping=True),
                                     nn_grid, X_train_s, y_train_s, X_test_s, y_test_s))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/model_results.csv", index=False)
    print("\nAll done. Results written to results/model_results.csv")
    print(results_df[["dataset", "model", "accuracy", "precision", "recall", "f1", "auc", "mcc"]])


if __name__ == "__main__":
    main()