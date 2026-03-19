"""
text_interface.py — Text CLI interface for the patient RAG system.
All requests go through the CoordinatorAgent (Mistral Small),
which routes to the appropriate action (segmentation, frame, rag_query).
Called by main.py in text mode.
"""

from RAG_Base.rag import PatientRAG
from coordinator.coordinator import CoordinatorAgent
from coordinator.session import SessionContext
from coordinator import handlers
from coordinator.catalog import get_catalog, format_catalog


def run_text_mode(rag: PatientRAG) -> None:
    """Interactive text-mode loop with coordinator agent."""

    coordinator = CoordinatorAgent()
    session = SessionContext()
    catalog = get_catalog(rag.collection)

    print("\nType 'quit' to exit, 'patients' to list available choices.")
    print("Type 'context' to see memorized session info.\n")

    while True:
        print("-" * 50)
        user_input = input("Vous : ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        # Navigation commands
        if user_input.lower() == "patients":
            catalog = get_catalog(rag.collection)
            print(f"\n{format_catalog(catalog, session.patient_id)}\n")
            continue

        if user_input.lower() in ("context", "contexte"):
            print(f"Session context: {session.summary()}")
            continue

        # ── Coordinateur ─────────────────────────────────────────────────────
        action = coordinator.route(user_input, session)

        # Champs manquants → afficher les choix disponibles + demande de clarification
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
                patient_id=action["patient_id"],
                accession_number=action["accession_number"],
                liaison_id=action["liaison_id"],
                view_type=action["view_type"],
            )
            print(f"\n{result}\n")

        elif route == "rag_query":
            if not action["patient_id"]:
                print("\n❌ Patient ID unknown. Please specify the patient in your message.\n")
                continue
            result = handlers.run_rag(
                rag=rag,
                patient_id=action["patient_id"],
                question=action["question"],
            )
            print(f"\n📋 Answer:\n{result}\n")

        else:
            print(f"[coordinator] Unknown route: {route}")
