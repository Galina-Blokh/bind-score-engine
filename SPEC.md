# Novella Bind Score — Technical Specification

Companion to [README.md](README.md). The README is the primary challenge deliverable; this file records implementation contracts.

## Paths (`src/paths.py`)

All default paths are resolved relative to the repository root (parent of `src/`):

| Variable | Default |
|----------|---------|
| `PROJECT_ROOT` | directory containing `pyproject.toml` |
| `DATA_DIR` | `PROJECT_ROOT / "data"` |
| `MODELS_DIR` | `PROJECT_ROOT / "models"` |
| `OUTPUT_DIR` | `PROJECT_ROOT / "output"` |
| `NOTEBOOKS_DIR` | `PROJECT_ROOT / "notebooks"` |

Override via CLI: `--data-dir`, `--models-dir`, `--output-dir`.

## Public API

```python
def feature(submission_id: int, t: int) -> dict[str, float]: ...
def bind_score(submission_id: int, t: int) -> float: ...
```

- `t ∈ {0, 7, 30}`
- `bind_score` returns P(bind) ∈ [0, 1]; requires `models/` from `main.py train`
- Uses `DATA_DIR` and `MODELS_DIR` from `src/paths.py` by default

## Constants (`src/config.py`)

See README § Design decisions for rationale.

## Data contract

| File | Columns |
|------|---------|
| `data/features_submissions.csv` | submissionId, createdDate, resolvedDate, agentEmail, label |
| `data/features_events.csv` | submissionId, event_type, event_date, email_char_count, email_attachment_count |

Event types: `EMAIL_INBOUND`, `EMAIL_OUTBOUND`, `QUOTE_RECEIVED`.

## Time logic

1. `T_ref = createdDate + t days`
2. Features use events with `event_date <= T_ref` (include pre-creation events)
3. F4 uses submissions with `resolvedDate < T_ref`
4. Target: final `label`

## Features

| Key | Definition |
|-----|------------|
| `inbound_recency_days` | Days since last inbound email ≤ T_ref; else `t + 30` |
| `quote_received_flag` | Any QUOTE_RECEIVED ≤ T_ref |
| `cumulative_email_density` | Σ chars + 100 × Σ attachments over events ≤ T_ref |
| `agent_smoothed_bind_rate` | Smoothed agent bind rate (α=10) using resolvedDate < T_ref |

## Modeling

- One model per horizon
- Split: 64% fit / 16% val / 20% test (temporal by createdDate)
- Hyperparameter search: RandomizedSearchCV + TimeSeriesSplit on fit, PR-AUC
- Model type selection: validation **PR-AUC**, tie-break precision @ top 20% (each candidate tuned once)
- Candidates: LogisticRegression, RandomForest, XGBoost
- Artifacts: `models/t_{0,7,30}/model.joblib`

## Evaluation

Metrics align with challenge.pdf **effort prioritization** (~15% bind rate). See README § Design decisions for full rationale.

| Metric | Use |
|--------|-----|
| **Precision @ top 20%** | Primary business metric; validation model-type selection (`TOP_K_FRACTION = 0.2`) |
| **PR-AUC** | Hyperparameter tuning, validation tie-break, permutation importance |
| **ROC-AUC** | Secondary test report only |

- Test metrics: ROC-AUC, PR-AUC, precision @ top 20% (reporting — not used for deployment)
- Task 2: permutation importance (10 repeats, PR-AUC), per horizon + overall mean
- Output: `output/evaluation_report.json`, `output/feature_importance_t*.png`

## CLI

Run from repository root:

```bash
uv run python main.py run
uv run python main.py train
uv run python main.py evaluate
uv run python main.py score --submission-id 1 --t 7
uv run pytest tests/ -q
uv run jupyter notebook notebooks/exploration.ipynb
```
