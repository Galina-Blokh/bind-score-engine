from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURE_NAMES, HORIZONS
from src.data_loader import load_data
from src.features import compute_features
from src.paths import DATA_DIR, MODELS_DIR


def _models_missing_message(models_dir: Path, t: int) -> str:
    return (
        f"No trained model for t={t} at {models_dir / f't_{t}' / 'model.joblib'}. "
        "Run: uv run python main.py train"
    )


def _load_bundle(models_dir: Path, t: int) -> dict:
    if t not in HORIZONS:
        raise ValueError(f"t must be one of {HORIZONS}, got {t}")
    path = models_dir / f"t_{t}" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(_models_missing_message(models_dir, t))
    return joblib.load(path)


def bind_score(submission_id: int, t: int) -> float:
    """Return P(bind) in [0, 1]; higher means more likely to sell.

    Requires trained models in models/ (run main.py train first).
    """
    return bind_score_with_paths(submission_id, t)


def bind_score_with_paths(
    submission_id: int,
    t: int,
    models_dir: Path | None = None,
    data_dir: Path | None = None,
    submissions: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> float:
    models_dir = models_dir or MODELS_DIR
    data_dir = data_dir or DATA_DIR
    if submissions is None or events is None:
        submissions, events = load_data(data_dir)

    bundle = _load_bundle(models_dir, t)
    feats = compute_features(submission_id, t, submissions, events)
    X = pd.DataFrame([[feats[name] for name in FEATURE_NAMES]], columns=list(FEATURE_NAMES))
    return float(bundle["pipeline"].predict_proba(X)[0, 1])
