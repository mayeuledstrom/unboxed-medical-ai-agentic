"""
orthanc.py — Study date lookup from local dict_accession_date.txt file.
Previously connected to Orthanc; now reads from a local file for offline use.

File format (Python dict literal):
    {
        '11092835': '20201125',
        '10492106': '20230823',
        ...
    }

Public API is unchanged: get_study_date_by_accession(), get_all_dates_for_patient(),
format_study_date() — callers do not need to be modified.
"""

import os
import ast
from dotenv import load_dotenv

RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(RAG_DIR, ".env"))

# Path to the accession→date mapping file (configurable via .env)
_UNBOXED_DIR = os.path.dirname(RAG_DIR)
DATE_FILE = os.getenv(
    "ACCESSION_DATE_FILE",
    os.path.join(_UNBOXED_DIR, "dict_accession_date.txt"),
)

# In-memory cache: accession_number -> study_date
_cache: dict[str, str | None] = {}
_loaded = False


def _load_date_file() -> None:
    """Loads the accession→date mapping file into the in-memory cache (once)."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    if not os.path.exists(DATE_FILE):
        print(f"[orthanc] ⚠️  Date file not found: {DATE_FILE}")
        return

    try:
        with open(DATE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # The file contains a Python dict literal — parse it safely
        data = ast.literal_eval(content)
        for acc, date in data.items():
            _cache[str(acc).strip()] = str(date).strip() if date else None
        print(f"[orthanc] ✅ Loaded {len(_cache)} accession dates from {DATE_FILE}")
    except Exception as e:
        print(f"[orthanc] ⚠️  Failed to load date file: {e}")


def get_study_date_by_accession(accession_number: str) -> str | None:
    """
    Returns the StudyDate (YYYYMMDD) for a given AccessionNumber.
    Returns None if not found.
    """
    _load_date_file()
    accession_number = str(accession_number).strip()
    return _cache.get(accession_number)


def get_all_dates_for_patient(accession_numbers: list[str]) -> dict[str, str | None]:
    """
    Returns a dict {accession_number: study_date} for a list of AccessionNumbers.
    """
    _load_date_file()
    return {
        str(acc).strip(): _cache.get(str(acc).strip())
        for acc in accession_numbers
    }


def format_study_date(raw_date: str | None) -> str:
    """
    Formats a DICOM StudyDate (YYYYMMDD) into a readable format (DD/MM/YYYY).
    Returns 'Unknown date' if None or invalid format.
    """
    if not raw_date or len(raw_date) != 8:
        return "Unknown date"
    try:
        return f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[:4]}"
    except Exception:
        return raw_date
