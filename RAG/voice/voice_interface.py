"""
voice_interface.py — Voice RAG interface with Voxtral 2.
Main loop: listen → transcribe → route → action → response.

The VoiceRouter handles navigation commands (quit, list_patients, switch_patient, calibrate).
The CoordinatorAgent handles medical requests and routes them to:
  - segmentation (patient + accession)
  - frame (accession + lesion + 2D/3D type)
  - rag_query (medical question via RAG)

The SessionContext maintains parameter persistence across the session.
Called by main.py in voice mode.
"""

import os
import sys
import subprocess

from RAG_Base.rag import PatientRAG
from voice.recorder import record_until_silence, calibrate_threshold, SILENCE_THRESHOLD
from voice.transcriber import VoxtralTranscriber
from voice.router import VoiceRouter
from coordinator.coordinator import CoordinatorAgent
from coordinator.session import SessionContext
from coordinator import handlers
from coordinator.catalog import get_catalog, format_catalog


def _get_available_patients(rag: PatientRAG) -> list[str]:
    """Retrieves the list of patients indexed in ChromaDB."""
    results = rag.collection.get(include=["metadatas"])
    return sorted(set(m["patient_id"] for m in results["metadatas"]))


def _handle_re_index() -> None:
    """Relaunches the data ingestion pipeline."""
    print("\n[voice] Re-indexing in progress...")
    ingest_path = os.path.join(os.path.dirname(__file__), "..", "RAG_Base", "ingest.py")
    python = sys.executable
    result = subprocess.run([python, ingest_path], capture_output=False)
    if result.returncode == 0:
        print("[voice] Re-indexing completed successfully.")
    else:
        print("[voice] Error during re-indexing.")


def run_voice_mode(rag: PatientRAG) -> None:
    """Interactive voice-mode loop (mic → Voxtral → router/coordinator → action)."""

    transcriber = VoxtralTranscriber(language="en")
    router = VoiceRouter()
    coordinator = CoordinatorAgent()
    session = SessionContext()
    catalog = get_catalog(rag.collection)

    # Default to first available patient
    patients = sorted(catalog.keys())
    if patients:
        session.patient_id = patients[0]

    voxtral_model = os.getenv("VOXTRAL_MODEL", "voxtral-mini-transcribe-2507")
    print(f"\nActive patient   : {session.patient_id}")
    print(f"Patients         : {patients}")
    print(f"Voxtral model    : {voxtral_model}")
    print(f"Silence threshold: {SILENCE_THRESHOLD} (type 'calibrate' to adjust)")
    print("\nPress [Enter] to speak, or type 'exit' to quit.\n")

    while True:
        user_input = input(f"\n[Patient: {session.patient_id}] Press Enter to speak > ").strip()

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if user_input.lower() in ("calibrate", "calibrer"):
            suggested = calibrate_threshold()
            print(f"Apply with: set SILENCE_THRESHOLD={suggested:.4f} in voice/recorder.py")
            continue

        if user_input.lower() in ("context", "contexte"):
            print(f"Session context: {session.summary()}")
            continue

        # ── Recording + transcription ────────────────────────────────────────
        print("\n🎤 Speak now...")
        audio_file = record_until_silence()
        text = transcriber.transcribe(audio_file)

        if not text:
            print("[voice] Nothing captured, please try again.")
            continue

        print(f"\n📝 Transcribed: \"{text}\"")

        # ── Voice router: navigation commands ────────────────────────────────
        intent = router.route(text)
        print(f"[router] Action detected: {intent.action}")

        if intent.action == "quit":
            print("Goodbye.")
            break

        elif intent.action == "help":
            print(f"\n{router.describe()}")

        elif intent.action == "list_patients":
            catalog = get_catalog(rag.collection)
            print(f"\n{format_catalog(catalog, session.patient_id)}\n")

        elif intent.action == "current_patient":
            print(f"\nActive patient: {session.patient_id}")

        elif intent.action == "switch_patient":
            new_pid = intent.params.get("patient_id")
            if new_pid:
                patients = _get_available_patients(rag)
                if new_pid in patients:
                    session.patient_id = new_pid
                    print(f"\n✅ Patient switched to: {session.patient_id}")
                else:
                    print(f"\n❌ Patient '{new_pid}' not found. Available: {patients}")
            else:
                print("\n❌ Patient identifier not recognized in your message.")

        elif intent.action == "re_index":
            _handle_re_index()
            rag = PatientRAG()
            catalog = get_catalog(rag.collection)
            patients = sorted(catalog.keys())
            print(f"[voice] Patients available after re-indexing: {patients}")

        elif intent.action == "empty":
            print("[voice] Empty command, please try again.")

        else:
            # ── Coordinator: medical requests (rag_query / segmentation / frame) ──
            action = coordinator.route(text, session)

            if action["missing"]:
                need_catalog = any(f in action["missing"] for f in ("patient_id", "accession_number"))
                if need_catalog:
                    catalog = get_catalog(rag.collection)
                    print(f"\n{format_catalog(catalog, session.patient_id, action['missing'])}\n")
                print(f"{action['clarification']}\n")
                continue

            route = action["route"]

            if route == "segmentation":
                result = handlers.run_segmentation(
                    patient_id=action["patient_id"],
                    accession_number=action["accession_number"],
                )
                print(f"\n{result}\n")

            elif route == "frame":
                result = handlers.run_frame(
                    accession_number=action["accession_number"],
                    liaison_id=action["liaison_id"],
                    view_type=action["view_type"],
                )
                print(f"\n{result}\n")

            elif route == "rag_query":
                if not action["patient_id"]:
                    print("\n❌ Patient ID unknown. Say 'patient [ID]' to select one.\n")
                    continue
                result = handlers.run_rag(
                    rag=rag,
                    patient_id=action["patient_id"],
                    question=action["question"],
                )
                print(f"\n📋 Answer:\n{result}\n")
