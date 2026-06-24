"""Repository path constants. All paths are derived from the project root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SUBMISSIONS_CSV = DATA_DIR / "features_submissions.csv"
EVENTS_CSV = DATA_DIR / "features_events.csv"
