"""
rag.py — PatientRAG class.
Ensures patient isolation via a ChromaDB filter on the `patient_id` metadata.
The LLM (Mistral Small) only receives documents from the requested patient in its context.
Documents are sorted chronologically and the prompt is selected dynamically
based on the detected study type (clinical_trial / nodule_control / assay).
"""

import re
import os
import sys
import chromadb
from mistralai import Mistral
from dotenv import load_dotenv
try:
    from RAG_Base.orthanc import format_study_date
    from RAG_Base.prompts import detect_study_type, get_prompt_for_type, STUDY_TYPE_LABELS
except ImportError:
    from orthanc import format_study_date
    from prompts import detect_study_type, get_prompt_for_type, STUDY_TYPE_LABELS

RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(RAG_DIR, ".env"))

# Path to the unboxed-hackathon root (for util_dicom import)
_UNBOXED_DIR = os.path.dirname(RAG_DIR)
if _UNBOXED_DIR not in sys.path:
    sys.path.insert(0, _UNBOXED_DIR)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

CHROMA_DIR = os.path.join(RAG_DIR, "RAG_Base", "chroma_db")
COLLECTION_NAME = "exams"
EMBED_MODEL = "mistral-embed"
LLM_MODEL = "mistral-small-latest"
TOP_K = 4  # Number of retrieved chunks
DICOM_DATASET_DIR = os.getenv("DICOM_DATASET_DIR", "/home/jovyan/work/dataset")

# Keywords triggering full report mode (both French and English user inputs)
_REPORT_KEYWORDS = [
    # French input keywords (users may type in French)
    r"\b(rapport|compte.?rendu|bilan|synth\u00e8se|synthese|r\u00e9sum\u00e9|resume|r\u00e9diger|rediger|g\u00e9n\u00e8re|genere|cr\u00e9er|creer|produire)\b",
    r"\b(r\u00e9ponse compl\u00e8te|rapport complet|compte rendu complet|fais.{0,10}rapport)\b",
    # English input keywords
    r"\b(report|summary|summarize|summarise|generate report|write report|full report|make report)\b",
]
_REPORT_RE = re.compile("|".join(_REPORT_KEYWORDS), re.IGNORECASE)


def _is_report_request(question: str) -> bool:
    """Returns True if the question explicitly requests a full report."""
    return bool(_REPORT_RE.search(question))


# Prompt for targeted questions (not a full report)
TARGETED_PROMPT_TEMPLATE = """You are a medical AI assistant helping clinicians review patient examination reports.
You have access to the patient's imaging reports from a {study_type_label} context.

IMPORTANT RULES:
- Answer ONLY the specific question asked. Do NOT generate a full structured report unless explicitly requested.
- Be concise and clinically precise. 2-5 sentences is usually sufficient for a targeted question.
- Use data from the provided context (measurements, dates, lesion names) to support your answer.
- Answer in the same language as the question.
- If the question cannot be answered from the provided context, say so clearly.
- You may mention temporal evolution only if directly relevant to the question."""


class PatientRAG:
    def __init__(self):
        self.mistral = Mistral(api_key=MISTRAL_API_KEY)

        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(
                f"ChromaDB database not found: {CHROMA_DIR}\n"
                "Please run first: python3 RAG_Base/ingest.py"
            )

        self.chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma.get_collection(COLLECTION_NAME)
        print(f"[RAG] ChromaDB loaded — {self.collection.count()} documents total")

    def _embed_query(self, text: str) -> list[float]:
        """Generates the embedding for the user query."""
        response = self.mistral.embeddings.create(
            model=EMBED_MODEL,
            inputs=[text],
        )
        return response.data[0].embedding

    def retrieve(self, patient_id: str, question: str) -> list[dict]:
        """
        Retrieves the most relevant documents for the given patient.
        The `where={"patient_id": patient_id}` filter ensures strict isolation:
        ChromaDB performs cosine search ONLY within the patient's document subset.
        """
        query_embedding = self._embed_query(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K, self.collection.count()),
            where={"patient_id": patient_id},  # ← STRICT ISOLATION
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": 1 - dist,  # cosine distance → similarity
                })
        return chunks

    def generate(self, patient_id: str, question: str, chunks: list[dict]) -> tuple[str, str]:
        """Generates a response via Mistral Small using only the patient's chunks.
        Automatically detects the study type and selects the matching specialized prompt.
        For full reports, also injects segmentation data (volume + dimensions per lesion).
        Returns (answer, study_type).
        """
        if not chunks:
            return (
                f"No data found for patient '{patient_id}' in the database. "
                "Please verify the patient identifier.",
                "unknown",
            )

        # Chronological sort: chunks with dates first, then undated
        def sort_key(c):
            d = c["metadata"].get("study_date", "")
            return d if d else "99999999"

        sorted_chunks = sorted(chunks, key=sort_key)

        # Detect study type from raw chunk texts
        report_texts = [c["text"] for c in sorted_chunks]
        study_type = detect_study_type(report_texts)

        is_report = _is_report_request(question)

        # Full report mode vs targeted question
        if is_report:
            system_prompt = get_prompt_for_type(study_type)
            print(f"[RAG] Mode: FULL REPORT ({STUDY_TYPE_LABELS.get(study_type, study_type)})")
        else:
            label = STUDY_TYPE_LABELS.get(study_type, study_type)
            system_prompt = TARGETED_PROMPT_TEMPLATE.format(study_type_label=label)
            print(f"[RAG] Mode: TARGETED QUESTION ({label})")

        context_parts = []
        for i, chunk in enumerate(sorted_chunks, 1):
            acc = chunk["metadata"].get("accession_number", "?")
            raw_date = chunk["metadata"].get("study_date", "")
            date_label = format_study_date(raw_date) if raw_date else "Unknown date"
            context_parts.append(
                f"--- Exam {i} | Date: {date_label} | AccessionNumber: {acc} ---\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        # For full reports: collect segmentation data for all accessions
        seg_section = ""
        if is_report:
            seg_section = self._build_segmentation_context(patient_id, sorted_chunks)

        user_message = (
            f"Patient ID: {patient_id}\n\n"
            f"CONTEXT (examinations of this patient only):\n{context}\n\n"
        )
        if seg_section:
            user_message += seg_section + "\n\n"
        user_message += f"QUESTION: {question}"

        response = self.mistral.chat.complete(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content, study_type

    @staticmethod
    def _build_segmentation_context(patient_id: str, sorted_chunks: list[dict]) -> str:
        """
        Runs the segmentation algorithm on each unique accession number found in sorted_chunks.
        Returns a formatted string section to inject into the LLM prompt.
        The section includes per-lesion volume and dimensions for each exam,
        and highlights volumetric changes between consecutive studies.
        """
        try:
            from util_dicom import get_info_lesions
        except ImportError:
            print("[RAG] ⚠️  util_dicom not available — segmentation data skipped")
            return ""

        # Collect unique accessions in chronological order
        seen = set()
        ordered_accessions = []
        for chunk in sorted_chunks:
            acc = chunk["metadata"].get("accession_number", "")
            raw_date = chunk["metadata"].get("study_date", "")
            if acc and acc not in seen:
                seen.add(acc)
                date_label = format_study_date(raw_date) if raw_date else "Unknown date"
                ordered_accessions.append((acc, date_label))

        if not ordered_accessions:
            return ""

        print(f"[RAG] Running segmentation on {len(ordered_accessions)} accession(s)…")

        # Run segmentation on each accession
        results = []  # list of (acc, date_label, lesions_list | None)
        for acc, date_label in ordered_accessions:
            try:
                lesions = get_info_lesions(
                    patientID=patient_id,
                    AccessionNumber=acc,
                    out_dir=DICOM_DATASET_DIR,
                )
                results.append((acc, date_label, lesions))
                print(f"[RAG]   ✅ {acc}: {len(lesions)} lesion(s) found")
            except Exception as e:
                print(f"[RAG]   ⚠️  {acc}: segmentation failed — {e}")
                results.append((acc, date_label, None))

        # Build the formatted section
        lines = ["SEGMENTATION DATA (automated 3D measurement results):"]
        lines.append("─" * 60)

        prev_lesions = None  # for computing volume evolution

        for acc, date_label, lesions in results:
            lines.append(f"\nExam: {acc} | Date: {date_label}")
            if lesions is None:
                lines.append("  [Segmentation unavailable for this exam]")
                prev_lesions = None
                continue

            if not lesions:
                lines.append("  No lesions detected.")
                prev_lesions = []
                continue

            for i, lesion in enumerate(lesions, 1):
                vol = lesion.get("volume", 0)
                dims = lesion.get("dimensions", (0, 0, 0))
                try:
                    vol_str = f"{float(vol):.1f} mm\u00b3"
                    dims_str = (
                        f"{float(dims[0]):.1f} \u00d7 {float(dims[1]):.1f} \u00d7 {float(dims[2]):.1f} mm"
                    )
                except (TypeError, ValueError, IndexError):
                    vol_str = str(vol)
                    dims_str = str(dims)

                # Compute volume change vs previous study (same lesion index)
                evolution_str = ""
                if prev_lesions is not None and i - 1 < len(prev_lesions):
                    prev_vol = prev_lesions[i - 1].get("volume", 0)
                    try:
                        delta = float(vol) - float(prev_vol)
                        pct = (delta / float(prev_vol) * 100) if float(prev_vol) != 0 else 0
                        arrow = "\u2197" if delta > 0 else ("\u2198" if delta < 0 else "\u2192")
                        evolution_str = f"  [{arrow} {delta:+.1f} mm\u00b3 ({pct:+.1f}% vs previous exam)]"
                    except (TypeError, ValueError):
                        pass

                lines.append(f"  Lesion {i}: Volume={vol_str} | Dimensions={dims_str}{evolution_str}")

            prev_lesions = lesions

        lines.append("\n" + "─" * 60)
        lines.append(
            "INSTRUCTIONS FOR REPORT GENERATION:\n"
            "The SEGMENTATION DATA above comes from automated 3D image analysis (objective measurements).\n"
            "The CONTEXT section above also contains clinical reports from previous studies, "
            "which may include lesion measurements recorded manually by radiologists "
            "(e.g. RECIST measurements, max diameter, lesion size in mm).\n\n"
            "In the report you generate, you MUST:\n"
            "1. For each exam/accession, present BOTH sets of measurements side by side:\n"
            "   - Automated 3D measurements (volume mm³, dimensions mm) from SEGMENTATION DATA\n"
            "   - Manual measurements from the clinical report text (if available)\n"
            "2. Comment on volumetric evolution between consecutive exams "
            "(growth ↗, regression ↘, stability →).\n"
            "3. Flag any lesion with >20% volume change between exams as clinically significant.\n"
            "4. If automated and manual measurements differ notably (>15%), explicitly note the "
            "discrepancy and indicate which source might be more reliable in context.\n"
            "5. Conclude with a summary of overall lesion burden evolution across all available studies."
        )
        return "\n".join(lines)

    def query(self, patient_id: str, question: str, verbose: bool = False) -> str:
        """Main entry point: retrieve + detect study type + generate."""
        patient_id = str(patient_id).strip()
        chunks = self.retrieve(patient_id, question)

        if verbose:
            print(f"\n[RAG] {len(chunks)} chunk(s) retrieved for patient '{patient_id}':")
            for c in chunks:
                raw_date = c["metadata"].get("study_date", "")
                date_str = format_study_date(raw_date) if raw_date else "no date"
                print(f"  - AccessionNumber={c['metadata']['accession_number']} | "
                      f"Date={date_str} | Similarity={c['similarity']:.3f}")

        answer, study_type = self.generate(patient_id, question, chunks)

        if verbose:
            label = STUDY_TYPE_LABELS.get(study_type, study_type)
            print(f"[RAG] Prompt used: {label}")

        return answer
