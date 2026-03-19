"""
prompts.py — Loading of specialized prompts and study type detection.
The type is inferred from the first lines of the patient's clinical reports.
"""

import os
import re

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# ── Prompt file loading ───────────────────────────────────────────────────

def _load_prompt_text(filename: str) -> str:
    """Loads a prompt file and extracts the Python variable value."""
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # The file defines a Python variable like: VAR = """..."""
    # We extract the content between triple-quotes
    match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


PROMPT_CLINICAL_TRIAL = _load_prompt_text("prompt_clinical_trial.txt")
PROMPT_NODULE_CONTROL = _load_prompt_text("prompt_nodule_control.txt")
PROMPT_ASSAY          = _load_prompt_text("prompt_assay.txt")

STUDY_TYPE_LABELS = {
    "clinical_trial": "Clinical Trial (RECIST)",
    "nodule_control": "Nodule Control / Surveillance",
    "assay":          "Biomarker / Assay",
}

# ── Keyword-based detection ─────────────────────────────────────────────────────
# Each study type is associated with a set of terms (case-insensitive).
# Score = number of terms found in the first N lines of patient reports.

_KEYWORDS: dict[str, list[str]] = {
    "clinical_trial": [
        "clinical trial", "essai clinique", "included in clinical trial",
        "recist", "target lesion", "non-target", "progressive disease",
        "partial response", "stable disease", "complete response",
        "oncology trial", "treatment arm", "randomized",
    ],
    "nodule_control": [
        "nodule", "nodule control", "nodule surveillance", "fleischner",
        "lung nodule", "pulmonary nodule", "follow-up nodule",
        "ground-glass", "part-solid", "solid nodule",
        "nodule follow", "suivi de nodule",
    ],
    "assay": [
        "assay", "biomarker", "pcr", "immunohistochemistry", "ihc",
        "elisa", "sequencing", "biopsy result", "cytology",
        "blood sample", "tissue sample", "mutation", "expression",
        "marker", "laboratory", "biomarqueur",
    ],
}

_HEADER_LINES = 10  # Number of lines to analyze in each report


def detect_study_type(report_texts: list[str]) -> str:
    """
    Infers the study type from the first lines of the patient's reports.
    Returns: 'clinical_trial' | 'nodule_control' | 'assay'
    If ambiguous or undetected, returns 'clinical_trial' as default.

    Args:
        report_texts: list of report texts (patient chunks)
    """
    # Extract first lines from each report
    header_text = ""
    for text in report_texts:
        lines = text.strip().splitlines()
        header_text += "\n".join(lines[:_HEADER_LINES]).lower() + "\n"

    scores: dict[str, int] = {t: 0 for t in _KEYWORDS}

    for study_type, keywords in _KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in header_text:
                scores[study_type] += 1

    best_type = max(scores, key=lambda t: scores[t])
    best_score = scores[best_type]

    # Debug log
    print(f"[prompts] Detection scores: {scores}")
    print(f"[prompts] Detected type: {best_type} (score={best_score})")

    # If no keyword found -> fallback to clinical_trial
    if best_score == 0:
        print("[prompts] No keyword detected, fallback -> clinical_trial")
        return "clinical_trial"

    return best_type


def get_prompt_for_type(study_type: str) -> str:
    """Returns the system prompt corresponding to the detected study type."""
    mapping = {
        "clinical_trial": PROMPT_CLINICAL_TRIAL,
        "nodule_control": PROMPT_NODULE_CONTROL,
        "assay":          PROMPT_ASSAY,
    }
    return mapping.get(study_type, PROMPT_CLINICAL_TRIAL)
