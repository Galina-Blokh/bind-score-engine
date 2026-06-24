import pandas as pd
import pytest

from src.config import ATTACHMENT_WEIGHT, HORIZONS, INBOUND_RECENCY_MISSING_DAYS
from src.data_loader import load_data
from src.features import compute_features, feature
from src.paths import DATA_DIR, MODELS_DIR


@pytest.fixture(scope="module")
def data():
    return load_data(DATA_DIR)


def test_feature_api_signature(data):
    submissions, events = data
    sid = int(submissions.iloc[0]["submissionId"])
    result = compute_features(sid, 0, submissions, events)
    assert set(result.keys()) == {
        "inbound_recency_days",
        "quote_received_flag",
        "cumulative_email_density",
        "agent_smoothed_bind_rate",
    }


def test_horizons_rejected(data):
    submissions, events = data
    sid = int(submissions.iloc[0]["submissionId"])
    with pytest.raises(ValueError, match="t must be one of"):
        compute_features(sid, 5, submissions, events)


def test_no_future_events_used(data):
    submissions, events = data
    sid = 1
    t = 0
    sub = submissions.loc[sid]
    t_ref = sub["createdDate"] + pd.Timedelta(days=t)
    feats = compute_features(sid, t, submissions, events)
    future = events[
        (events["submissionId"] == sid) & (events["event_date"] > t_ref)
    ]
    allowed = events[
        (events["submissionId"] == sid) & (events["event_date"] <= t_ref)
    ]
    expected_density = float(
        allowed["email_char_count"].sum()
        + ATTACHMENT_WEIGHT * allowed["email_attachment_count"].sum()
    )
    assert feats["cumulative_email_density"] == expected_density
    if not future.empty:
        assert feats["cumulative_email_density"] != float(
            events[events["submissionId"] == sid]["email_char_count"].sum()
            + ATTACHMENT_WEIGHT * events[events["submissionId"] == sid]["email_attachment_count"].sum()
        )


def test_quote_flag_respects_cutoff(data):
    submissions, events = data
    sid = int(submissions.iloc[0]["submissionId"])
    feats_early = compute_features(sid, 0, submissions, events)
    feats_late = compute_features(sid, 30, submissions, events)
    assert feats_late["quote_received_flag"] >= feats_early["quote_received_flag"]


def test_inbound_recency_missing_imputation(data):
    submissions, events = data
    no_inbound = None
    for sid in submissions["submissionId"]:
        sub = submissions.loc[sid]
        t_ref = sub["createdDate"]
        inbound = events[
            (events["submissionId"] == sid)
            & (events["event_type"] == "EMAIL_INBOUND")
            & (events["event_date"] <= t_ref)
        ]
        if inbound.empty:
            no_inbound = int(sid)
            break
    assert no_inbound is not None
    feats = compute_features(no_inbound, 0, submissions, events)
    assert feats["inbound_recency_days"] == float(INBOUND_RECENCY_MISSING_DAYS)


def test_feature_public_api(data):
    sid = int(data[0].iloc[0]["submissionId"])
    result = feature(sid, HORIZONS[0])
    assert "agent_smoothed_bind_rate" in result


def test_bind_score_requires_models():
    from pathlib import Path

    from src.bind_score import bind_score_with_paths

    missing_models = Path("/nonexistent/models")
    with pytest.raises(FileNotFoundError, match="Run:"):
        bind_score_with_paths(1, 7, models_dir=missing_models, data_dir=DATA_DIR)
