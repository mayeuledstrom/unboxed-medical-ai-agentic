"""
voice_main.py — Compatibility alias.
Launches the voice interface as before.

Legacy usage (preserved):
    python3 voice_main.py

Preferred usage:
    python3 main.py --mode voice
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RAG_Base.rag import PatientRAG
from voice.voice_interface import run_voice_mode


BANNER = """
╔══════════════════════════════════════════════════════╗
║     Medical RAG — Voice Assistant                    ║
║     Speak your command after the [mic] signal        ║
╚══════════════════════════════════════════════════════╝
"""


def main() -> None:
    print(BANNER)
    print("Loading RAG knowledge base...")
    try:
        rag = PatientRAG()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    run_voice_mode(rag)


if __name__ == "__main__":
    main()
