"""
router.py — Voice intent router.
Parses transcribed text from Voxtral and detects the action to execute.
Normalizes punctuation and accents before matching to be robust
to transcription variations (Voxtral may produce unexpected forms).
"""

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class Intent:
    """Represents a detected intent."""
    action: str                        # action name
    params: dict = field(default_factory=dict)  # extracted parameters
    raw_text: str = ""                 # raw transcribed text


# ── Voice route definitions ──────────────────────────────────────────────────
# IMPORTANT: patterns are applied on NORMALIZED text (no accents,
# no punctuation, lowercase). Ex: "Stop it!" → "stop it"

ROUTES = {
    # Stop the assistant
    "quit": [
        r"\b(quit|quitter|exit|stop|goodbye|bye|farewell|end|terminate|done)\b",
    ],

    # Help
    "help": [
        r"\b(help|commands?|what can you do|how does (it|this) work|instructions?|usage)\b",
    ],

    # List patients — strict patterns to avoid false positives
    "list_patients": [
        r"\b(list|show|display|get).{0,10}(patients?|subjects?|cases?)\b",
        r"\b(which|what|available)\s+(patients?|subjects?|cases?)\b",
        r"\bpatients?\s+(available|existing|list)\b",
        r"^patients?$",
    ],

    # Current patient — BEFORE switch_patient
    "current_patient": [
        r"\b(current|active|selected|chosen)\s+patient\b",
        r"\bwhich\s+patient.{0,15}(active|current|selected)\b",
        r"\b(who is the patient|current patient|what patient)\b",
    ],

    # Re-indexing — BEFORE switch_patient (to avoid ID false positives)
    "re_index": [
        r"\b(reindex|re-index|rebuild|reprocess|reingest)\b",
        r"\b(update|refresh).{0,15}(database|index|data|knowledge base)\b",
        r"\b(reload|ingest).{0,10}(data|documents?|base)\b",
    ],

    # Patient switch — compact ID OR spelled out
    "switch_patient": [
        # Compact alphanumeric ID (e.g. 063F6BB9)
        r"\b(patient|subject|case)\s+([A-Z0-9]{6,12})\b",
        r"\b(switch|change|select|use|go to|set).{0,20}patient.{0,5}([A-Z0-9]{6,12})\b",
        # Spelled out: "patient zero six three f six b b nine"
        r"\b(patient|subject|case)\s+(zero|one|two|three|four|five|six|seven|eight|nine|[a-f])\b",
    ],
}


def _normalize(text: str) -> str:
    """
    Normalizes text before routing:
    - Lowercase
    - Remove accents (NFD + Mn filter)
    - Remove punctuation
    """
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Spelled-out digits/letters → ID character
_SPELL_MAP: dict[str, str] = {
    # English digits
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Hex letters A-F (most common in DICOM IDs)
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E", "f": "F",
    "g": "G", "h": "H", "i": "I", "j": "J", "k": "K", "l": "L",
    "m": "M", "n": "N", "o": "O", "p": "P", "q": "Q", "r": "R",
    "s": "S", "t": "T", "u": "U", "v": "V", "w": "W", "x": "X",
    "y": "Y", "z": "Z",
}


def _spell_to_id(tokens: list[str]) -> str:
    """
    Reconstructs an ID from a list of spelled tokens.
    Ex: ["zero", "six", "three", "f", "six", "b", "b", "nine"] → "063F6BB9"
    Ignores unrecognized words ("patient", "the", etc.).
    """
    result = ""
    for token in tokens:
        t = token.lower().strip()
        if t in _SPELL_MAP:
            result += _SPELL_MAP[t]
        elif len(t) == 1 and t.isalnum():
            result += t.upper()
    return result


class VoiceRouter:
    """
    Detects the intent from transcribed text.
    Specific intents take priority over the default RAG query fallback.
    """

    def __init__(self):
        # Compile patterns once
        self._compiled: dict[str, list[re.Pattern]] = {
            action: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
            for action, patterns in ROUTES.items()
        }

    def route(self, text: str) -> Intent:
        """
        Analyzes text and returns the matching Intent.
        Normalizes first (accents, punctuation, case) to be robust
        to Voxtral transcription variations.
        If no route matches → action='rag_query' (passed to CoordinatorAgent).
        """
        text_clean = text.strip()
        text_norm = _normalize(text_clean)

        print(f"[router] Normalized: '{text_norm}'")

        if not text_norm:
            return Intent(action="empty", raw_text=text_clean)

        # Check each route in priority order
        for action, patterns in self._compiled.items():
            for pattern in patterns:
                m = pattern.search(text_norm)
                if m:
                    params = self._extract_params(action, text_norm, m)
                    return Intent(action=action, params=params, raw_text=text_clean)

        # No route matched → pass to CoordinatorAgent
        return Intent(action="rag_query", params={"question": text_clean}, raw_text=text_clean)

    def _extract_params(self, action: str, text: str, match: re.Match) -> dict:
        """Extracts action-specific parameters.
        text is the NORMALIZED text (lowercase, no accents).
        """
        if action == "switch_patient":
            # Attempt 1: compact alphanumeric ID with at least 1 digit
            pid_match = re.search(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{6,12}\b", text, re.IGNORECASE)
            if pid_match:
                return {"patient_id": pid_match.group(0).upper()}

            # Attempt 2: spelled-out ID token by token after the keyword 'patient'
            # Ex: "patient zero six three f six b b nine" → "063F6BB9"
            after_kw = re.split(r"\b(patients?|subject|case)\b", text, maxsplit=1)
            if len(after_kw) >= 3:
                rest = after_kw[-1].strip()
                tokens = rest.split()
                candidate = _spell_to_id(tokens)
                if len(candidate) >= 4:  # minimal ID length
                    print(f"[router] ID reconstructed from spelled tokens: '{candidate}'")
                    return {"patient_id": candidate}

        return {}

    def describe(self) -> str:
        """Returns the list of available voice commands."""
        return """
Available voice commands:
  - [medical question]          → RAG query on the active patient
  - "patient [ID]"              → Switch active patient
  - "list patients"             → Show available patients
  - "current patient"           → Show selected patient
  - "reindex"                   → Update the knowledge base
  - "help"                      → Show this help
  - "quit" / "stop"             → Exit the assistant
""".strip()
