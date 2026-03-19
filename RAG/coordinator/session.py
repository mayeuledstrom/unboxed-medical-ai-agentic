"""
session.py — Persistent session context between conversation turns.
Stores already-provided parameters to avoid asking for them again.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionContext:
    """
    Maintains the current state of the conversation.
    All fields are optional and updated incrementally as the conversation progresses.

    Persistent fields:
      patient_id        — active patient identifier
      accession_number  — current DICOM accession number
      liaison_id        — selected lesion/structure number (for the frame route)
      view_type         — view type: 0 = 2D, 1 = 3D
      pending_route     — route waiting for parameters (e.g. "segmentation")
                          Set back to None once the route is executed.
    """

    patient_id: Optional[str] = None
    accession_number: Optional[str] = None
    liaison_id: Optional[int] = None
    view_type: Optional[int] = None  # 0 = 2D, 1 = 3D
    pending_route: Optional[str] = None
    pending_question: Optional[str] = None  # stores the original RAG question during clarification

    def update(self, data: dict) -> None:
        """
        Updates non-None fields from the dict `data`.
        Fields absent or None in `data` do not overwrite existing values.
        """
        if data.get("patient_id"):
            self.patient_id = str(data["patient_id"]).strip()
        if data.get("accession_number"):
            self.accession_number = str(data["accession_number"]).strip()
        if data.get("liaison_id") is not None:
            try:
                self.liaison_id = int(data["liaison_id"])
            except (ValueError, TypeError):
                pass
        if data.get("view_type") is not None:
            try:
                vt = int(data["view_type"])
                if vt in (0, 1):
                    self.view_type = vt
            except (ValueError, TypeError):
                pass

    def summary(self) -> str:
        """Short text summary of the current session context (for logs)."""
        view_label = {0: "2D", 1: "3D"}.get(self.view_type, "?") if self.view_type is not None else None
        parts = []
        if self.patient_id:
            parts.append(f"patient={self.patient_id}")
        if self.accession_number:
            parts.append(f"accession={self.accession_number}")
        if self.liaison_id is not None:
            parts.append(f"lesion={self.liaison_id}")
        if self.view_type is not None:
            parts.append(f"view={view_label}")
        return "{" + ", ".join(parts) + "}" if parts else "{empty}"
