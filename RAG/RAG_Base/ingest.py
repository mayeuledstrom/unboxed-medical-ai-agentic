"""
ingest.py — Indexing of exams.xlsx into ChromaDB.
Each document is tagged with `patient_id` in metadata to ensure patient isolation.
Uses mistral-embed for embeddings.
Retrieves study dates from Orthanc for temporal analysis.
"""

import os
import sys
import hashlib
import pandas as pd
import chromadb
from chromadb.config import Settings
from mistralai import Mistral
from dotenv import load_dotenv
try:
    from RAG_Base.orthanc import get_all_dates_for_patient, format_study_date
except ImportError:
    from orthanc import get_all_dates_for_patient, format_study_date

# Path to the .env at the RAG root directory
RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(RAG_DIR, ".env"))

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

DATA_PATH = os.path.join(RAG_DIR, "Data", "exams.xlsx")
CHROMA_DIR = os.path.join(RAG_DIR, "RAG_Base", "chroma_db")

EMBED_MODEL = "mistral-embed"
COLLECTION_NAME = "exams"


def build_document_text(row: pd.Series, study_date: str | None = None) -> str:
    """Builds the text of a document from the relevant columns."""
    parts = []

    patient_id = row.get("PatientID", "")
    accession = row.get("AccessionNumber", "")
    report_col = "Clinical information data (Pseudo reports)"
    report = row.get(report_col, "")
    lesion_size = row.get("lesion size in mm", "")
    serie = row.get("Série avec les masques de DICOM SEG", "")

    if patient_id:
        parts.append(f"Patient ID: {patient_id}")
    if accession:
        parts.append(f"Accession Number: {accession}")
    if study_date:
        parts.append(f"Study Date: {format_study_date(study_date)} (raw: {study_date})")
    if lesion_size:
        parts.append(f"Lesion size: {lesion_size} mm")
    if serie:
        parts.append(f"DICOM SEG Series: {serie}")
    if report and str(report).strip():
        parts.append(f"Report:\n{str(report).strip()}")

    return "\n".join(parts)


def get_embeddings(client: Mistral, texts: list[str]) -> list[list[float]]:
    """Calls the Mistral API to get embeddings for a list of texts."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        inputs=texts,
    )
    return [item.embedding for item in response.data]


def ingest():
    print(f"[ingest] Reading {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    print(f"[ingest] {len(df)} exams found, {df['PatientID'].nunique()} distinct patients")

    # Initialize ChromaDB (persistent on disk)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present (clean re-indexing)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"[ingest] Existing collection deleted, full re-indexing")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Fetch Orthanc dates in batch (single list call)
    all_accessions = [str(row.get("AccessionNumber", "")).strip() for _, row in df.iterrows()]
    print(f"[ingest] Fetching study dates from Orthanc…")
    dates_by_accession = get_all_dates_for_patient(all_accessions)
    found = sum(1 for d in dates_by_accession.values() if d is not None)
    print(f"[ingest] Dates found: {found}/{len(all_accessions)}")

    # Build documents
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        patient_id = str(row.get("PatientID", "unknown")).strip()
        accession = str(row.get("AccessionNumber", idx)).strip()
        study_date = dates_by_accession.get(accession)

        text = build_document_text(row, study_date=study_date)

        # Unique stable ID
        doc_id = hashlib.md5(f"{patient_id}_{accession}".encode()).hexdigest()

        documents.append(text)
        metadatas.append({
            "patient_id": patient_id,
            "accession_number": accession,
            "study_date": study_date or "",       # metadata for chronological sorting
            "source": "exams.xlsx",
        })
        ids.append(doc_id)

    # Compute embeddings (single batch)
    print(f"[ingest] Computing embeddings via {EMBED_MODEL}…")
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    embeddings = get_embeddings(mistral_client, documents)

    # Index into ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"[ingest] ✅ {len(documents)} documents indexed in ChromaDB ({CHROMA_DIR})")
    print(f"[ingest] Indexed patients: {sorted(set(m['patient_id'] for m in metadatas))}")


if __name__ == "__main__":
    ingest()
