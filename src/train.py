"""Temporal training: tune with PR-AUC, deploy by validation ranking.

Each candidate is tuned once on the fit split (TimeSeriesSplit + PR-AUC). Model type
is chosen on the validation split: highest val PR-AUC, tie-break val P@20 (P@20 alone
is too noisy on ~16% of data). The winner is refit on fit+val and saved to models/.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    FEATURE_NAMES,
    HORIZONS,
    RANDOM_STATE,
    TOP_K_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION_OF_TRAIN,
)
from src.data_loader import load_data
from src.evaluate import evaluate_predictions
from src.features import build_feature_frame
from src.paths import DATA_DIR, MODELS_DIR


def temporal_split(
    submissions: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[set[int], set[int]]:
    """Earliest train_fraction submissions → train; remainder → test."""
    ordered = submissions.sort_values("createdDate")
    cutoff = int(len(ordered) * train_fraction)
    train_ids = set(ordered.iloc[:cutoff]["submissionId"])
    test_ids = set(ordered.iloc[cutoff:]["submissionId"])
    return train_ids, test_ids


def temporal_train_val_split(
    submissions: pd.DataFrame,
    train_ids: set[int],
    val_fraction_of_train: float = VAL_FRACTION_OF_TRAIN,
) -> tuple[set[int], set[int]]:
    """Split train-period submissions into fit (earlier) and val (later)."""
    train_subs = submissions[submissions["submissionId"].isin(train_ids)].sort_values("createdDate")
    val_start = int(len(train_subs) * (1 - val_fraction_of_train))
    fit_ids = set(train_subs.iloc[:val_start]["submissionId"])
    val_ids = set(train_subs.iloc[val_start:]["submissionId"])
    return fit_ids, val_ids


def selection_key(val_metrics: dict[str, float]) -> tuple[float, float]:
    """Lexicographic rank for model type selection on the validation split."""
    return (val_metrics["pr_auc"], val_metrics["precision_at_top_20"])


def _model_candidates() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_leaf": [2, 5],
            },
        ),
        "xgboost": (
            Pipeline(
                [
                    (
                        "model",
                        XGBClassifier(
                            random_state=RANDOM_STATE,
                            eval_metric="logloss",
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            {
                "model__n_estimators": [100, 150],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
            },
        ),
    }


def _sorted_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values("createdDate")


def _param_grid_size(param_grid: dict) -> int:
    return len(list(ParameterGrid(param_grid)))


def _hyperparameter_search(
    fit_rows: pd.DataFrame,
    pipeline: Pipeline,
    param_grid: dict,
) -> GridSearchCV | RandomizedSearchCV:
    fit_rows = _sorted_rows(fit_rows)
    X_fit = fit_rows[list(FEATURE_NAMES)]
    y_fit = fit_rows["label"]
    n_splits = min(3, max(2, len(fit_rows) // 50))
    cv = TimeSeriesSplit(n_splits=n_splits)
    search_kwargs = {
        "estimator": pipeline,
        "scoring": "average_precision",
        "cv": cv,
        "n_jobs": -1,
    }
    if _param_grid_size(param_grid) <= 8:
        search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            **search_kwargs, param_grid=param_grid
        )
    else:
        search = RandomizedSearchCV(
            **search_kwargs,
            param_distributions=param_grid,
            n_iter=8,
            random_state=RANDOM_STATE,
        )
    search.fit(X_fit, y_fit)
    return search


def _train_horizon_candidates(
    fit_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> tuple[list[dict], str, Pipeline, dict, dict]:
    """Tune each candidate once; select on validation; refit winner on fit+val."""
    X_val = val_rows[list(FEATURE_NAMES)]
    y_val = val_rows["label"].values
    X_test = test_rows[list(FEATURE_NAMES)]
    y_test = test_rows["label"].values
    sorted_train = _sorted_rows(train_rows)

    candidate_rows: list[dict] = []
    best_name = ""
    best_refit: Pipeline | None = None
    best_val_metrics: dict = {}
    best_test_metrics: dict = {}
    best_key = (-1.0, -1.0)

    for name, (pipeline, param_grid) in _model_candidates().items():
        search = _hyperparameter_search(fit_rows, pipeline, param_grid)
        tuned = search.best_estimator_
        val_scores = tuned.predict_proba(X_val)[:, 1]
        val_metrics = evaluate_predictions(y_val, val_scores)

        refit = clone(tuned)
        refit.fit(sorted_train[list(FEATURE_NAMES)], sorted_train["label"])
        test_scores = refit.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_predictions(y_test, test_scores)

        key = selection_key(val_metrics)
        row = {
            "model": name,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_precision_at_top_20": val_metrics["precision_at_top_20"],
            **test_metrics,
            "cv_pr_auc": float(search.best_score_),
            "deployed": False,
        }
        candidate_rows.append(row)

        if key > best_key:
            best_key = key
            best_name = name
            best_refit = refit
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics

    assert best_refit is not None
    for row in candidate_rows:
        if row["model"] == best_name:
            row["deployed"] = True

    return candidate_rows, best_name, best_refit, best_val_metrics, best_test_metrics


def train_models(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    data_dir = data_dir or DATA_DIR
    models_dir = models_dir or MODELS_DIR
    submissions, events = load_data(data_dir)
    feature_frame = build_feature_frame(submissions, events)
    train_ids, test_ids = temporal_split(submissions)
    fit_ids, val_ids = temporal_train_val_split(submissions, train_ids)

    models_dir.mkdir(parents=True, exist_ok=True)
    ordered = submissions.sort_values("createdDate")
    train_cutoff = ordered.iloc[len(train_ids) - 1]["createdDate"].isoformat()
    val_cutoff = ordered[ordered["submissionId"].isin(val_ids)]["createdDate"].min()
    val_cutoff_iso = val_cutoff.isoformat() if pd.notna(val_cutoff) else train_cutoff

    training_report: dict = {
        "train_fraction": TRAIN_FRACTION,
        "val_fraction_of_train": VAL_FRACTION_OF_TRAIN,
        "train_submissions": len(train_ids),
        "fit_submissions": len(fit_ids),
        "val_submissions": len(val_ids),
        "test_submissions": len(test_ids),
        "train_cutoff_created_date": train_cutoff,
        "val_cutoff_created_date": val_cutoff_iso,
        "model_selection_metric": (
            f"validation PR-AUC, tie-break precision_at_top_{int(TOP_K_FRACTION * 100)}"
        ),
        "hyperparameter_cv": "TimeSeriesSplit on fit period (submission order)",
        "horizons": {},
        "all_model_results": [],
    }

    for t in HORIZONS:
        rows_t = feature_frame[feature_frame["t"] == t]
        fit_rows = rows_t[rows_t["submissionId"].isin(fit_ids)]
        val_rows = rows_t[rows_t["submissionId"].isin(val_ids)]
        train_rows = rows_t[rows_t["submissionId"].isin(train_ids)]
        test_rows = rows_t[rows_t["submissionId"].isin(test_ids)]

        candidate_rows, best_name, best_refit, best_val_metrics, best_test_metrics = (
            _train_horizon_candidates(fit_rows, val_rows, train_rows, test_rows)
        )

        for row in candidate_rows:
            training_report["all_model_results"].append({"t": t, **row})

        horizon_dir = models_dir / f"t_{t}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": best_refit,
            "model_name": best_name,
            "horizon": t,
            "feature_names": list(FEATURE_NAMES),
            "train_cutoff_created_date": train_cutoff,
            "global_bind_rate": float(submissions["label"].mean()),
        }
        joblib.dump(bundle, horizon_dir / "model.joblib")

        training_report["horizons"][str(t)] = {
            "best_model": best_name,
            "val_metrics": best_val_metrics,
            "test_metrics": best_test_metrics,
        }

    report_path = models_dir / "training_report.json"
    report_path.write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    return training_report
