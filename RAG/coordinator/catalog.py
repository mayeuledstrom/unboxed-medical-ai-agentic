"""
catalog.py — Reads the patient/accession number catalog from ChromaDB.
Displays available choices when this information is requested.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb


def get_catalog(collection: "chromadb.Collection") -> dict[str, list[str]]:
    """
    Returns a dict { patient_id: [accession_number, ...] }
    by reading all metadata from the ChromaDB collection.
    Accession numbers are sorted alphabetically per patient.
    """
    results = collection.get(include=["metadatas"])
    catalog: dict[str, set[str]] = {}

    for meta in results["metadatas"]:
        pid = meta.get("patient_id", "?")
        acc = meta.get("accession_number", "?")
        catalog.setdefault(pid, set()).add(acc)

    # Sort for stable display
    return {pid: sorted(accs) for pid, accs in sorted(catalog.items())}


def format_catalog(
    catalog: dict[str, list[str]],
    current_patient: str | None = None,
    missing_fields: list[str] | None = None,
) -> str:
    """
    Formats the catalog as readable text for CLI/voice display.

    Example output:
        Available patients and accession numbers:
        ┌─ P001 (active patient)
        ├── ACC001
        └── ACC002
        ┌─ P002
        └── ACC003

    If `missing_fields` contains only 'accession_number' and
    `current_patient` is known, only shows the active patient's
    accession numbers (shorter list).
    """
    missing_fields = missing_fields or []

    # Case: only accession number missing and patient is known → short list
    if (
        "accession_number" in missing_fields
        and "patient_id" not in missing_fields
        and current_patient
        and current_patient in catalog
    ):
        accs = catalog[current_patient]
        lines = [f"Available accession numbers for patient {current_patient}:"]
        for acc in accs:
            lines.append(f"  • {acc}")
        return "\n".join(lines)

    # General case: display full catalog
    lines = ["Available patients and accession numbers:"]
    for pid, accs in catalog.items():
        marker = " (active patient)" if pid == current_patient else ""
        if accs:
            lines.append(f"  ┌─ {pid}{marker}")
            for i, acc in enumerate(accs):
                prefix = "  └──" if i == len(accs) - 1 else "  ├──"
                lines.append(f"  {prefix} {acc}")
        else:
            lines.append(f"  • {pid}{marker} (no exams)")
    return "\n".join(lines)
