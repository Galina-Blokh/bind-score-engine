from pathlib import Path

import pandas as pd

from src.paths import DATA_DIR, EVENTS_CSV, SUBMISSIONS_CSV

SUBMISSIONS_FILE = SUBMISSIONS_CSV.name
EVENTS_FILE = EVENTS_CSV.name


def summarize_missing(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Count nulls and empty strings per column (raw CSV view)."""
    rows: list[dict] = []
    n = len(df)
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        empty_count = 0
        if df[col].dtype == object:
            empty_count = int(df[col].fillna("").astype(str).str.strip().eq("").sum())
        rows.append(
            {
                "table": table_name,
                "column": col,
                "rows": n,
                "null_count": null_count,
                "null_pct": round(null_count / n, 4) if n else 0.0,
                "empty_string_count": empty_count,
                "dtype": str(df[col].dtype),
            }
        )
    return pd.DataFrame(rows)


def audit_raw_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Missing-value report on CSV files before imputation."""
    data_dir = data_dir or DATA_DIR
    submissions_raw = pd.read_csv(data_dir / SUBMISSIONS_FILE)
    events_raw = pd.read_csv(data_dir / EVENTS_FILE)
    return pd.concat(
        [
            summarize_missing(submissions_raw, "submissions"),
            summarize_missing(events_raw, "events"),
        ],
        ignore_index=True,
    )


def audit_loaded_data(
    submissions: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Missing-value report after load_submissions / load_events cleaning."""
    return pd.concat(
        [
            summarize_missing(submissions.reset_index(drop=True), "submissions_clean"),
            summarize_missing(events, "events_clean"),
        ],
        ignore_index=True,
    )


def load_submissions(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DATA_DIR
    df = pd.read_csv(data_dir / SUBMISSIONS_FILE)
    df["createdDate"] = pd.to_datetime(df["createdDate"])
    df["resolvedDate"] = pd.to_datetime(df["resolvedDate"])
    df["agentEmail"] = df["agentEmail"].fillna("unknown")
    df["label"] = df["label"].astype(int)
    df = df.set_index("submissionId", drop=False)
    return df


def load_events(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DATA_DIR
    df = pd.read_csv(data_dir / EVENTS_FILE)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["email_char_count"] = pd.to_numeric(df["email_char_count"], errors="coerce").fillna(0)
    df["email_attachment_count"] = pd.to_numeric(
        df["email_attachment_count"], errors="coerce"
    ).fillna(0)
    return df


def load_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or DATA_DIR
    return load_submissions(data_dir), load_events(data_dir)
