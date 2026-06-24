from __future__ import annotations

import pandas as pd

from src.config import (
    ATTACHMENT_WEIGHT,
    FEATURE_NAMES,
    HORIZONS,
    INBOUND_RECENCY_MISSING_DAYS,
    SMOOTHING_ALPHA,
)
from src.data_loader import load_data


def _events_at_horizon(
    submission_id: int,
    t_ref: pd.Timestamp,
    events: pd.DataFrame,
) -> pd.DataFrame:
    return events[
        (events["submissionId"] == submission_id) & (events["event_date"] <= t_ref)
    ]


def _global_bind_rate_before(
    t_ref: pd.Timestamp,
    submissions: pd.DataFrame,
) -> float:
    prior = submissions[submissions["resolvedDate"] < t_ref]
    if prior.empty:
        return float(submissions["label"].mean())
    return float(prior["label"].mean())


def _agent_smoothed_bind_rate(
    agent_email: str,
    t_ref: pd.Timestamp,
    submissions: pd.DataFrame,
    alpha: float = SMOOTHING_ALPHA,
) -> float:
    global_rate = _global_bind_rate_before(t_ref, submissions)
    prior = submissions[
        (submissions["agentEmail"] == agent_email) & (submissions["resolvedDate"] < t_ref)
    ]
    binds = prior["label"].sum()
    total = len(prior)
    return float((binds + alpha * global_rate) / (total + alpha))


def compute_features(
    submission_id: int,
    t: int,
    submissions: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, float]:
    if t not in HORIZONS:
        raise ValueError(f"t must be one of {HORIZONS}, got {t}")

    if submission_id not in submissions["submissionId"].values:
        raise KeyError(f"submission_id {submission_id} not found")

    sub = submissions.loc[submission_id]
    t_ref = sub["createdDate"] + pd.Timedelta(days=t)
    evts = _events_at_horizon(submission_id, t_ref, events)

    inbound = evts[evts["event_type"] == "EMAIL_INBOUND"]
    if inbound.empty:
        inbound_recency_days = float(t + INBOUND_RECENCY_MISSING_DAYS)
    else:
        inbound_recency_days = (t_ref - inbound["event_date"].max()).total_seconds() / 86400

    quote_received_flag = float((evts["event_type"] == "QUOTE_RECEIVED").any())
    cumulative_email_density = float(
        evts["email_char_count"].sum() + ATTACHMENT_WEIGHT * evts["email_attachment_count"].sum()
    )
    agent_smoothed_bind_rate = _agent_smoothed_bind_rate(
        sub["agentEmail"], t_ref, submissions
    )

    return {
        "inbound_recency_days": inbound_recency_days,
        "quote_received_flag": quote_received_flag,
        "cumulative_email_density": cumulative_email_density,
        "agent_smoothed_bind_rate": agent_smoothed_bind_rate,
    }


def feature(submission_id: int, t: int) -> dict[str, float]:
    """Calculate features for a submission at horizon t (challenge.pdf API)."""
    submissions, events = load_data()
    return compute_features(submission_id, t, submissions, events)


def feature_with_data(
    submission_id: int,
    t: int,
    submissions: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, float]:
    """Same as feature() but with pre-loaded data (used by training pipeline)."""
    return compute_features(submission_id, t, submissions, events)


def build_feature_frame(
    submissions: pd.DataFrame,
    events: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    rows: list[dict] = []
    for submission_id in submissions["submissionId"]:
        label = int(submissions.loc[submission_id, "label"])
        created = submissions.loc[submission_id, "createdDate"]
        for t in horizons:
            feats = compute_features(submission_id, t, submissions, events)
            rows.append(
                {
                    "submissionId": submission_id,
                    "t": t,
                    "createdDate": created,
                    "label": label,
                    **feats,
                }
            )
    return pd.DataFrame(rows)
