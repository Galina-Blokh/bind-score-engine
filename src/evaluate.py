"""Held-out test metrics and permutation importance.

Metrics match broker prioritization: P@20 = conversion rate in the
top-scored fraction (TOP_K_FRACTION); PR-AUC = imbalanced-class ranking for tuning
and feature importance. See README § Design decisions and notebooks §3.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import FEATURE_NAMES, HORIZONS, RANDOM_STATE, TOP_K_FRACTION


def precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.2) -> float:
    if len(y_true) == 0:
        return 0.0
    n_top = max(1, int(np.ceil(len(y_true) * k)))
    order = np.argsort(y_score)[::-1]
    top = y_true[order[:n_top]]
    return float(top.mean())


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "precision_at_top_20": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision_at_top_20": precision_at_top_k(y_true, y_score, k=TOP_K_FRACTION),
    }


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
    )
    return (
        pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def load_model_bundle(models_dir: Path, t: int) -> dict:
    path = models_dir / f"t_{t}" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model for t={t} at {path}")
    return joblib.load(path)


def run_evaluation(
    feature_frame: pd.DataFrame,
    test_ids: set[int],
    models_dir: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict] = []
    importance_by_t: dict[int, list[dict]] = {}

    for t in HORIZONS:
        bundle = load_model_bundle(models_dir, t)
        model = bundle["pipeline"]
        model_name = bundle["model_name"]

        test_rows = feature_frame[(feature_frame["t"] == t) & (feature_frame["submissionId"].isin(test_ids))]
        X_test = test_rows[list(FEATURE_NAMES)]
        y_test = test_rows["label"].values
        scores = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, scores)
        comparison_rows.append({"model": model_name, "t": t, **metrics})

        train_rows = feature_frame[
            (feature_frame["t"] == t) & (~feature_frame["submissionId"].isin(test_ids))
        ]
        X_train = train_rows[list(FEATURE_NAMES)]
        y_train = train_rows["label"]
        importance = compute_permutation_importance(model, X_train, y_train)
        importance_by_t[t] = importance.to_dict(orient="records")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(importance["feature"], importance["importance_mean"], xerr=importance["importance_std"])
        ax.invert_yaxis()
        ax.set_xlabel("Permutation importance (PR-AUC drop)")
        ax.set_title(f"Feature importance at t={t} ({model_name})")
        fig.tight_layout()
        fig.savefig(output_dir / f"feature_importance_t{t}.png", dpi=120)
        plt.close(fig)

    overall = (
        pd.DataFrame(importance_by_t[HORIZONS[0]])
        .drop(columns=["importance_std"])
        .rename(columns={"importance_mean": f"importance_t_{HORIZONS[0]}"})
    )
    for t in HORIZONS[1:]:
        t_df = pd.DataFrame(importance_by_t[t])[["feature", "importance_mean"]].rename(
            columns={"importance_mean": f"importance_t_{t}"}
        )
        overall = overall.merge(t_df, on="feature")
    mean_cols = [f"importance_t_{t}" for t in HORIZONS]
    overall["importance_mean"] = overall[mean_cols].mean(axis=1)
    overall = overall.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    overall_ranking = overall[["feature", "importance_mean"]].to_dict(orient="records")

    report = {
        "model_comparison": comparison_rows,
        "feature_importance": importance_by_t,
        "feature_importance_overall": overall_ranking,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
