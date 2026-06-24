"""Project-wide constants. Values and rationale are documented in README § Design decisions."""

# Scoring horizons (days since submission creation)
HORIZONS = (0, 7, 30)

# Temporal split: earliest 80% train, latest 20% held-out test (no random shuffling).
TRAIN_FRACTION = 0.8
# Within train, the latest 20% of train-period submissions form a validation set for model
# selection. Fit = 64%, val = 16%, test = 20% of all submissions.
VAL_FRACTION_OF_TRAIN = 0.2

RANDOM_STATE = 42

# F1: penalize missing inbound email with horizon + buffer (approx. one month silence).
INBOUND_RECENCY_MISSING_DAYS = 30

# F3: scale attachments to a similar magnitude as character counts (median char ~300,
# median attachments ~1 in early events → weight 100).
ATTACHMENT_WEIGHT = 100.0

# F4: Bayesian smoothing toward global bind rate for agents with little history.
SMOOTHING_ALPHA = 10

# Broker prioritization metric: conversion rate among top-scored 20% of submissions.
TOP_K_FRACTION = 0.2

FEATURE_NAMES = (
    "inbound_recency_days",
    "quote_received_flag",
    "cumulative_email_density",
    "agent_smoothed_bind_rate",
)
