"""
config.py — Central, portable path configuration.

Everything else in this project imports paths from here instead of
hardcoding them, so the whole repo runs unchanged after a `git clone`
on any machine: just `pip install -r requirements.txt` and run.

Override the data directory with the KESTREL_DATA_DIR environment
variable if you keep the CSVs somewhere else.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KESTREL_DATA_DIR", PROJECT_ROOT / "data"))

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_unlabelled.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
PARTNERS_PATH = DATA_DIR / "partners.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
OPS_POLICY_PATH = DATA_DIR / "ops-policy.pdf"
EMAIL_THREAD_PATH = DATA_DIR / "email-thread.txt"

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def require_data_files():
    """Fail fast with a clear message if the data/ folder isn't populated."""
    missing = [p for p in [TRAIN_PATH, TEST_PATH, PARTNERS_PATH, PRODUCTS_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data file(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\n\nExpected them in: {DATA_DIR}\n"
            "Place the Kestrel data pack CSVs there (see data/README.txt in the pack), "
            "or set KESTREL_DATA_DIR to point elsewhere."
        )
