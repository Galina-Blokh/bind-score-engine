import pandas as pd
import pytest

from src.data_loader import audit_loaded_data, audit_raw_data, load_data
from src.paths import DATA_DIR


def test_raw_audit_finds_quote_email_nulls():
    report = audit_raw_data(DATA_DIR)
    quote_char_nulls = report[
        (report["table"] == "events")
        & (report["column"] == "email_char_count")
    ]["null_count"].iloc[0]
    assert quote_char_nulls > 0


def test_clean_audit_has_no_nulls_after_load():
    submissions, events = load_data(DATA_DIR)
    report = audit_loaded_data(submissions, events)
    assert report["null_count"].sum() == 0


def test_agent_email_nulls_filled():
    submissions, _ = load_data(DATA_DIR)
    assert submissions["agentEmail"].isna().sum() == 0
