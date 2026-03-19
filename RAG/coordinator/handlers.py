"""
handlers.py — Action executors routed by the coordinator.

  - run_segmentation(patient_id, accession_number)             → real segmentation via util_dicom
  - run_frame(patient_id, accession_number, liaison_id, type)  → real frame display via agent2D/3D
  - run_rag(rag, patient_id, question)                 → actual PatientRAG call
"""

from __future__ import annotations
import os
import sys
from typing import TYPE_CHECKING
from dotenv import load_dotenv

if TYPE_CHECKING:
    from RAG_Base.rag import PatientRAG

# ── Environment ───────────────────────────────────────────────────────────────
_RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_RAG_DIR, ".env"))

# Path to the local DICOM dataset directory (configurable via .env)
DICOM_DATASET_DIR = os.getenv("DICOM_DATASET_DIR", "/home/jovyan/work/dataset")

# Add the parent of unboxed-hackathon to sys.path so util_dicom can be imported
_UNBOXED_DIR = os.path.dirname(_RAG_DIR)
if _UNBOXED_DIR not in sys.path:
    sys.path.insert(0, _UNBOXED_DIR)


# ── Segmentation ──────────────────────────────────────────────────────────────

def run_segmentation(patient_id: str, accession_number: str) -> str:
    """
    Runs the segmentation algorithm on a patient exam using util_dicom.get_info_lesions().

    Expects DICOM data at:
        {DICOM_DATASET_DIR}/{patient_id}/{accession_number}/

    Returns a human-readable summary of detected lesions (volume + dimensions).
    """
    print(f"\n[handlers] 🔬 SEGMENTATION — patient={patient_id} | accession={accession_number}")
    print(f"[handlers]   Dataset dir: {DICOM_DATASET_DIR}")

    try:
        from util_dicom import get_info_lesions
    except ImportError as e:
        return (
            f"❌ Could not import util_dicom: {e}\n"
            f"   Make sure the unboxed-hackathon directory is accessible at: {_UNBOXED_DIR}"
        )

    try:
        lesions = get_info_lesions(
            patientID=patient_id,
            AccessionNumber=accession_number,
            out_dir=DICOM_DATASET_DIR,
        )
    except Exception as e:
        return (
            f"❌ Segmentation failed for patient '{patient_id}' / accession '{accession_number}':\n"
            f"   {type(e).__name__}: {e}"
        )

    if not lesions:
        return (
            f"⚠️  No lesions detected for patient '{patient_id}' / accession '{accession_number}'.\n"
            f"   Check that the DICOM data is present at: {DICOM_DATASET_DIR}/{patient_id}/{accession_number}/"
        )

    # Format the results
    lines = [
        f"🔬 Segmentation complete — Patient: {patient_id} | Accession: {accession_number}",
        f"   {len(lesions)} lesion(s) detected:\n",
    ]
    for i, lesion in enumerate(lesions, 1):
        volume = lesion.get("volume", "N/A")
        dims = lesion.get("dimensions", ("N/A", "N/A", "N/A"))
        try:
            volume_str = f"{float(volume):.1f} mm³"
        except (TypeError, ValueError):
            volume_str = str(volume)
        try:
            dims_str = f"{float(dims[0]):.1f} × {float(dims[1]):.1f} × {float(dims[2]):.1f} mm"
        except (TypeError, ValueError, IndexError):
            dims_str = str(dims)
        lines.append(f"   Lesion {i}:")
        lines.append(f"     Volume    : {volume_str}")
        lines.append(f"     Dimensions: {dims_str}")

    return "\n".join(lines)


# ── Frame display ─────────────────────────────────────────────────────────────

def run_frame(patient_id: str, accession_number: str, liaison_id: int, view_type: int) -> str:
    """
    Displays a 2D or 3D frame of an exam for a given lesion.
    Uses agent2D.frame2D (view_type=0) or agent3D.frame3D (view_type=1) imported dynamically.

    Parameters:
      patient_id       — Patient UUID
      accession_number — DICOM exam identifier
      liaison_id       — lesion/structure number to display (1-based index)
      view_type        — 0 = 2D (slice), 1 = 3D (volume rendering)
    """
    view_label = "3D" if view_type == 1 else "2D"
    print(f"\n[handlers] 🖼️  FRAME {view_label} — patient={patient_id} | accession={accession_number} | lesion={liaison_id}")

    # For arrays, map 1-based (user) to 0-based index
    lesion_idx = max(0, liaison_id - 1)

    try:
        if view_type == 0:
            from agent2D import frame2D
            result = frame2D(
                lesionID=lesion_idx,
                patientID=patient_id,
                AccessionNumber=accession_number,
                source_folder=DICOM_DATASET_DIR,
                output_folder="/home/jovyan/work/output"
            )
            return result
        else:
            from agent3D import frame3D
            result = frame3D(
                lesionId=lesion_idx,
                patientID=patient_id,
                AccessionNumber=accession_number,
                src_dir=DICOM_DATASET_DIR,
                out_dir="/home/jovyan/work/output"
            )
            return result
    except ImportError as e:
        return (
            f"❌ Could not import agent{view_label}: {e}\n"
            f"   Make sure agent2D.py and agent3D.py are accessible."
        )
    except Exception as e:
        return (
            f"❌ Frame display failed for patient '{patient_id}' / lesion '{liaison_id}':\n"
            f"   {type(e).__name__}: {e}"
        )


# ── RAG query ─────────────────────────────────────────────────────────────────

def run_rag(rag: "PatientRAG", patient_id: str, question: str) -> str:
    """Calls the RAG system to answer a medical question about a patient."""
    print(f"\n[handlers] 📄 RAG — patient={patient_id} | question=\"{question}\"")
    answer = rag.query(patient_id=patient_id, question=question, verbose=True)
    return answer
