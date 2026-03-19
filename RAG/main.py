"""
main.py — Single entry point for the medical RAG assistant.
Offers a text mode (keyboard CLI) and a voice mode (Voxtral 2 + microphone).

Usage:
    python3 main.py                  # interactive menu to choose mode
    python3 main.py --mode text      # launch text mode directly
    python3 main.py --mode voice     # launch voice mode directly
"""

import sys
import os
import argparse

# Allow importing RAG_Base and voice from the RAG root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RAG_Base.rag import PatientRAG


BANNER = """
╔══════════════════════════════════════════════════════╗
║        Medical RAG — Mistral Small                   ║
╚══════════════════════════════════════════════════════╝
"""

MODE_MENU = """
  Choose your interaction mode:
    [1] Text mode   — keyboard CLI
    [2] Voice mode  — microphone + Voxtral 2
    [q] Quit
"""


def _select_mode() -> str:
    """Shows the mode selection menu and returns 'text' or 'voice'."""
    while True:
        print(MODE_MENU)
        choice = input("Mode > ").strip().lower()
        if choice in ("1", "text"):
            return "text"
        if choice in ("2", "voice"):
            return "voice"
        if choice in ("q", "quit", "exit"):
            print("Goodbye.")
            sys.exit(0)
        print("Invalid choice, enter 1, 2 or q.")


def main() -> None:
    # ── CLI argument parsing ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Medical RAG — text or voice interface"
    )
    parser.add_argument(
        "--mode",
        choices=["text", "voice"],
        default=None,
        help="Interface mode: 'text' (keyboard CLI) or 'voice' (mic + Voxtral 2). "
             "If omitted, an interactive menu is shown at startup.",
    )
    args = parser.parse_args()

    print(BANNER)

    # Mode selection (menu or argument)
    mode = args.mode if args.mode else _select_mode()

    # ── Load RAG once ────────────────────────────────────────────────────────
    print("\nLoading knowledge base…")
    try:
        rag = PatientRAG()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    # ── Dispatch to selected interface ───────────────────────────────────────
    if mode == "text":
        print("\n▶  Text mode active\n")
        from RAG_Base.text_interface import run_text_mode
        run_text_mode(rag)
    else:
        print("\n▶  Voice mode active\n")
        from voice.voice_interface import run_voice_mode
        run_voice_mode(rag)


if __name__ == "__main__":
    main()
