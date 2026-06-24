# Bind Score Engine

Bind-score model for E&S insurance submissions. Given `(submission_id, t)` — where `t` is days since creation — the engine returns a **probability of binding** in `[0, 1]`. Higher scores mean the submission is more likely to sell, so brokers can prioritize effort on the highest-ranked queue.

## 1. Repository layout

```
bind-score-engine/
├── data/
│   ├── features_submissions.csv   # 881 submissions, label = sold (1) or not (0)
│   └── features_events.csv        # 17,211 lifecycle events
├── src/
│   ├── config.py                  # documented constants (split ratios, feature params)
│   ├── paths.py                   # PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUTPUT_DIR
│   ├── data_loader.py             # CSV loading and typing
│   ├── features.py                # feature(submission_id, t)
│   ├── bind_score.py              # bind_score(submission_id, t)
│   ├── train.py                   # temporal splits + model training
│   └── evaluate.py                # metrics + permutation importance
├── notebooks/
│   └── exploration.ipynb          # end-to-end walkthrough (EDA → scoring)
├── tests/
│   ├── test_features.py           # leakage checks + API tests
│   ├── test_data_loader.py
│   └── test_train.py              # model-selection key tests
├── models/                        # saved by train (one best model per t)
├── output/                        # evaluation report + importance charts
├── main.py                        # CLI entrypoint
├── SPEC.md                        # technical specification (companion doc)
├── README.md
├── pyproject.toml
└── uv.lock
```

## 2. How to run

Requires **Git Bash** and [uv](https://docs.astral.sh/uv/) with Python 3.11.

### Full pipeline (recommended first run)

Run from the repository root (Git Bash):

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv run python main.py run
```

This runs **train → evaluate → sample score** in order. `bind_score()` requires trained models in `models/`.

Optional: override directories with env vars or CLI flags (defaults come from `src/paths.py`):

```bash
uv run python main.py run --data-dir data --models-dir models
```

### Step-by-step

```bash
uv run python main.py train
uv run python main.py evaluate
uv run python main.py score --submission-id 1 --t 7
uv run pytest tests/ -q
uv run jupyter notebook notebooks/exploration.ipynb
```

| Command | Purpose |
|---------|---------|
| `run` | Full pipeline: train, evaluate, print sample `bind_score` |
| `train` | Temporal split, tune 3 model types per horizon, save best model per `t` |
| `evaluate` | Held-out test metrics + permutation-importance charts |
| `score` | Return `bind_score(submission_id, t)` for one submission |
| Notebook | `notebooks/exploration.ipynb` — EDA, features, training, evaluation, scoring |

### Public API

```python
from src.features import feature
from src.bind_score import bind_score

feature(submission_id, t)     # dict of 4 features; t ∈ {0, 7, 30}
bind_score(submission_id, t)  # P(bind) in [0, 1]; requires prior train step
```

---

## 3. Design decisions

This section documents **why** each major choice was made.

### Problem framing

| Decision | Rationale |
|----------|-----------|
| **Bind score = predicted probability** | Higher score ⇒ more likely to bind. Classifier `predict_proba` is calibrated for ranking and interpretable as a sale probability. |
| **Separate model per horizon (t=0, 7, 30)** | Feature distributions shift over the submission lifecycle (e.g. quotes absent at t=0). One model per `t` avoids forcing a single function to handle all stages. |
| **Final `label` as target at all horizons** | Predict bind outcome from early signals. Using the ultimate sold/not-sold label is standard; all 881 submissions in the dataset are resolved. |

### Data and leakage prevention

| Decision | Rationale |
|----------|-----------|
| **Centralized paths in `src/paths.py`** | `PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, and `OUTPUT_DIR` are derived from the repo layout — no machine-specific paths in code or docs. |
| **`T_ref = createdDate + t days`** | `t` is days since creation. All features use only data on or before `T_ref`. |
| **Include pre-creation events (471 rows)** | Events are keyed by `submissionId` and often reflect email thread activity before the formal submission timestamp. Excluding them would discard real signal. |
| **Agent stats use `resolvedDate < T_ref` only** | Agent bind rate (F4) must not use outcomes that were not yet known at scoring time. |
| **Actual CSV column names** | Data uses `event_date` / `QUOTE_RECEIVED`; code follows the data. |

### Feature engineering constants (`src/config.py`)

| Constant | Value | Rationale |
|----------|-------|-----------|
| F1 missing inbound imputation | `t + 30` days | No inbound email implies disengagement; impute a penalizing recency rather than zero. |
| F3 attachment weight | `100` | Median email ≈ hundreds of chars, attachments ≈ 0–3. Weight puts attachments on a comparable scale to character counts. |
| F4 smoothing α | `10` | 397 unique agents; raw rates are noisy. Shrink toward global bind rate (~14.8%) until enough agent history exists. |

### Train / validation / test protocol

| Split | Fraction | Rationale |
|-------|----------|-----------|
| **Fit** | 64% (earliest submissions) | Hyperparameter tuning on past data only. |
| **Validation** | 16% (next submissions) | Model **type** selection without touching the test set. |
| **Test** | 20% (latest submissions) | Single held-out report of generalization to future submissions. |

All splits are **temporal by `createdDate`** — no random shuffling (sales patterns drift over time).

| Decision | Rationale |
|----------|-----------|
| **Hyperparameter CV: `TimeSeriesSplit` on fit period** | Avoids random k-fold leakage across time within the fit window. |
| **Model selection: validation PR-AUC, tie-break P@20** | Broker **prioritization** is the product goal. Val P@20 alone is unstable on ~140 submissions (top-20% ≈ 29 rows); at t=0 it picked LR (17% val P@20) while RF had better val ranking and test P@20. Selection uses **val PR-AUC** first (stable, matches tuning objective), then val P@20. Each candidate is tuned **once**; the saved artifact is the same refit pipeline reported in the comparison table. |
| **Report all models on test set** | Full comparison table for transparency; test metrics are for reporting only, not selection. |

### Evaluation metrics

| Metric | Role |
|--------|------|
| **Precision @ top 20%** | Primary **business** metric on test — simulates “work the top-ranked queue first”. Tie-break for model type selection on validation. |
| **PR-AUC** | Hyperparameter tuning; **primary** model-type selection on validation (~15% bind rate). |
| **ROC-AUC** | Secondary ranking quality across all thresholds. |

### Feature significance

**Permutation importance** (10 repeats, PR-AUC scoring) shuffles each feature and measures performance drop. Preferred over tree default impurity because it reflects predictive contribution on the fitted model and is comparable across feature types. Reported **per horizon** and as an **overall mean** across t ∈ {0, 7, 30}.

---

## 4. Results

**Dataset:** 881 submissions, **14.8% bind rate** (130 sold), median time-to-resolve **21 days**.

**Split dates:** train cutoff **2020-06-16**, validation from **2020-05-22**, test = submissions after train cutoff.

### Features

Four features at **t ∈ {0, 7, 30}**:

| ID | Feature | Definition at T_ref |
|----|---------|---------------------|
| F1 | `inbound_recency_days` | Days since most recent `EMAIL_INBOUND` ≤ T_ref; if none → `t + 30`. |
| F2 | `quote_received_flag` | 1 if any `QUOTE_RECEIVED` event ≤ T_ref, else 0. |
| F3 | `cumulative_email_density` | Σ `email_char_count` + 100 × Σ `email_attachment_count` over events ≤ T_ref. |
| F4 | `agent_smoothed_bind_rate` | Agent bind rate from submissions with `resolvedDate < T_ref`, smoothed (α=10) toward global rate. |

**Why these four:** They cover engagement velocity (F1), underwriting milestone (F2), information intensity (F3), and broker quality (F4) — all computable from email, quote, and agent fields available at scoring time.

### Feature significance

Method: permutation importance on the deployed model per horizon.

**Overall ranking (mean across t = 0, 7, 30)**

| Rank | Feature | Mean importance |
|------|---------|-----------------|
| 1 | agent_smoothed_bind_rate | 0.304 |
| 2 | cumulative_email_density | 0.101 |
| 3 | quote_received_flag | 0.074 |
| 4 | inbound_recency_days | 0.066 |

**t = 0** (Logistic Regression): agent (0.016) > email density (0.010) > quote (0.000) > inbound (−0.005)

**t = 7** (Random Forest): agent (0.354) > email density (0.168) > inbound (0.126) > quote (0.115)

**t = 30** (Random Forest): agent (0.541) > email density (0.127) > quote (0.107) > inbound (0.077)

Charts: `output/feature_importance_t{0,7,30}.png`

**Interpretation:** Agent history dominates overall. Lifecycle features (quote, email volume) gain importance after the first week when those events have occurred.

### Model comparison & scoring

Deployed models selected on **validation PR-AUC** (tie-break val P@20), evaluated on held-out **test** set.

**Model comparison (test set — all candidates trained; bold = deployed per horizon)**

| Model | t | ROC-AUC | PR-AUC | Precision @ Top 20% |
|-------|---|---------|--------|---------------------|
| **Logistic Regression** | **0** | 0.567 | 0.244 | **22.2%** |
| Random Forest | 0 | 0.733 | 0.356 | 33.3% |
| XGBoost | 0 | 0.727 | 0.305 | 36.1% |
| Logistic Regression | 7 | 0.788 | 0.404 | 38.9% |
| **Random Forest** | **7** | **0.801** | **0.431** | **36.1%** |
| XGBoost | 7 | 0.695 | 0.402 | 25.0% |
| Logistic Regression | 30 | 0.936 | 0.672 | 61.1% |
| **Random Forest** | **30** | **0.946** | **0.708** | **61.1%** |
| XGBoost | 30 | 0.929 | 0.671 | 58.3% |

**Note:** At t=0, Random Forest / XGBoost achieve higher **test** P@20 than the deployed Logistic Regression, but LR had the best **validation PR-AUC** (0.146 vs 0.101 / 0.060). Test metrics are for reporting only — we never select on the test set.

**Early vs late:** Deployed test P@20 rises from **22.2%** (t=0) to **61.1%** (t=30) vs **14.8%** baseline — early scoring still ~1.5× baseline, supporting prioritization with limited data.

**Example:** `bind_score(1, 7) = 0.4958` (after training)

---

## 5. Exploration notebook

[`notebooks/exploration.ipynb`](notebooks/exploration.ipynb) is the narrative walkthrough of this repo — same pipeline as the CLI, with plots and commentary:

1. **EDA** — submissions, events, bind rate
2. **Features** — four time-bounded features at `t ∈ {0, 7, 30}`
3. **Training** — temporal split, model comparison, evaluation metrics
4. **Feature significance** — permutation importance by horizon
5. **Scoring** — `bind_score(submission_id, t)` API
6. **Challenges & future work** — limitations and path to production

Run the first code cell before any other (sets up `sys.path` and reloads `src.*`). Artifacts written by the notebook match the CLI: `models/t_{0,7,30}/model.joblib`, `models/training_report.json`, `output/evaluation_report.json`, `output/feature_importance_t*.png`.

---

## 6. Challenges

Limitations and open issues in the current implementation:

| Area | Challenge |
|------|-----------|
| **Data scale** | 881 submissions, ~15% bind rate — small sample; validation P@20 is noisy (~29 submissions in the top-20% slice). |
| **Early horizon** | t=0 has weak signal (test P@20 ~1.5× baseline); brokers get limited lift on day of creation. |
| **Feature set** | Only four hand-engineered features from email/quote/agent fields; no line-of-business, geography, seasonality, or submission content. |
| **Temporal drift** | Single chronological split; patterns may shift — no rolling retrain or drift monitoring yet. |
| **Model selection** | Val metrics do not always align with test (especially at t=0); no nested CV or larger holdout for stable deployment picks. |
| **Probabilities** | `bind_score` is raw `predict_proba` — not calibrated; ranked order is the main product, but absolute probabilities may mislead brokers. |
| **Leakage discipline** | F4 and event cutoffs are handled carefully, but production needs automated leakage tests on every feature change. |
| **Operational gap** | Batch notebook/CLI only — no online scoring service, feature store, or broker-facing UI integration. |

---

## 7. Future work

**Serving & integration**
- REST/gRPC scoring API (`bind_score(submission_id, t)`) with latency SLOs, auth, and versioning per horizon
- Real-time feature pipeline: ingest events as they arrive, materialize features at T_ref without batch CSV reload
- Broker workflow hook: ranked queue in CRM, explain-why snippet per submission (top features + recency)

**Model quality**
- Richer features: LOB, premium band, carrier, seasonality, email NLP/sentiment, quote turnaround time
- Probability calibration (isotonic / Platt) on a dedicated calibration set; report reliability diagrams
- Stable model selection: nested temporal CV, larger validation window, or optimize blended metric (val PR-AUC + val P@20)
- Ensemble or monotonic constraints if business rules require interpretable ranking
- Continuous retrain on rolling windows; champion/challenger and shadow deployment before cutover

**MLOps & monitoring**
- Feature store + model registry (e.g. MLflow); pinned artifacts per environment
- Monitor: bind rate, score distribution, P@20 on recent cohorts, PR-AUC drift, feature null rates (F1 spikes at t+30)
- Alerts when production P@20 falls below baseline or val/test gap widens
- Automated leakage and temporal-integrity tests in CI (extend `tests/`)

**Governance & product**
- Document broker-facing definition of the score; A/B test prioritization vs current manual triage
- Fairness review across agents/regions; audit trail for scores shown to users
- Feedback loop: log broker actions (worked / skipped / bound) to refine labels and features over time

An **ideal** system scores every open submission within seconds of new events, ranks an actionable top-K queue with calibrated win probabilities, retrains safely on fresh data, and proves lift in broker conversion — not just offline AUC.
