"""
transcriber.py — Audio transcription via Voxtral Mini V2.
Uses the same Mistral client as the RAG — no need for a separate client.
"""

import os
from mistralai import Mistral
from dotenv import load_dotenv

RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(RAG_DIR, ".env"))

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

# Voxtral model — configurable via .env if the name changes
VOXTRAL_MODEL = os.getenv("VOXTRAL_MODEL", "voxtral-v2-mini-2507")


class VoxtralTranscriber:
    """Transcribes a WAV audio file to text via the Mistral Voxtral API."""

    def __init__(self, language: str = "en"):
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.language = language
        print(f"[transcriber] VoxtralTranscriber initialized (model={VOXTRAL_MODEL}, language={language})")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes the WAV file at audio_path.
        Returns the transcribed text, or "" on error.
        Deletes the temporary file after transcription.
        """
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            response = self.client.audio.transcriptions.complete(
                model=VOXTRAL_MODEL,
                file={"file_name": os.path.basename(audio_path), "content": audio_bytes},
                language=self.language,
            )
            text = response.text.strip()
            print(f"[transcriber] Transcription: '{text}'")
            return text

        except Exception as e:
            print(f"[transcriber] Transcription error: {e}")
            return ""
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except Exception:
                pass
