"""
coordinator.py — Mistral Small coordinator agent.
Analyzes the user request and routes to the appropriate action:
  - segmentation : run the segmentation algorithm (patient + accession)
  - frame        : display a 2D/3D frame (accession + lesion + type)
  - rag_query    : medical question via RAG (patient + question)

Uses SessionContext for parameter persistence across conversation turns.
If required parameters are missing, requests a clarification.
"""

import json
import os
import re
from mistralai import Mistral
from dotenv import load_dotenv

from coordinator.session import SessionContext

RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(RAG_DIR, ".env"))

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LLM_MODEL = "mistral-small-latest"

# ── System prompt du coordinateur ────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a medical imaging AI coordinator. Your sole task is to analyze the user's message and output a JSON object — nothing else.

## Available routes

### 1. segmentation
Triggered when the user wants to run a segmentation algorithm on a patient exam.
Keywords: "segment", "segmentation", "run segmentation", "launch segmentation", "analyze", "automatic analysis", "detect lesions".

Required fields: patient_id, accession_number

### 2. frame
Triggered when the user wants to display a 2D or 3D frame/view/slice of an exam.
Keywords: "show", "display", "view", "frame", "slice", "2D", "3D", "render", "visualize", "image".

Required fields: accession_number, liaison_id (integer — lesion/structure number), view_type (0=2D, 1=3D)

### 3. rag_query
Triggered for any medical question about a patient's reports, exams, history, results, measurements, etc.
This is the default route for everything that is not segmentation or frame display.

Required fields: patient_id, question

## CRITICAL RULE — liaison_id extraction
For the `frame` route, `liaison_id` is the lesion or structure number the user refers to.
Examples:
- "lesion 2" → liaison_id: 2
- "the 2nd lesion" → liaison_id: 2
- "structure number 3" → liaison_id: 3
- "lesion number 2" → liaison_id: 2
- "2" (when pending_route is frame) → liaison_id: 2
ALWAYS extract the integer even if it is the only token in the message.

## CRITICAL RULE — pending_route
If the context contains `pending_route: segmentation` or `pending_route: frame`, the user
is answering a clarification request for that action.
You MUST set `route` to the value of `pending_route` — do NOT re-route to rag_query.

## Output format

Always respond with ONLY a valid JSON object. No explanation, no text outside the JSON.

```json
{
  "route": "segmentation" | "frame" | "rag_query",
  "patient_id": "<string or null>",
  "accession_number": "<string or null>",
  "liaison_id": <integer or null>,
  "view_type": <0 or 1 or null>,
  "question": "<string or null>",
  "missing": ["<field_name>", ...]
}
```

Rules:
- Set `missing` to the list of required fields absent from BOTH the message AND the provided context.
- If a field is already provided in the context, do NOT add it to `missing`.
- If no fields are missing, set `missing` to [].
- Extract patient_id, accession_number, liaison_id, view_type from the message when present.
- For `question` in rag_query route, copy the user's full original message.
- For view_type: "3D" → 1, "2D" → 0, default for frame = 0 (2D).
"""


class CoordinatorAgent:
    """
    Coordinator agent: receives a user request + session context,
    calls Mistral Small to route and extract parameters,
    merges with persistent context,
    returns a dict describing the action to execute.
    """

    def __init__(self):
        self.mistral = Mistral(api_key=MISTRAL_API_KEY)

    def route(self, user_message: str, session: SessionContext) -> dict:
        """
        Analyzes `user_message` and returns an action dict.
        If session.pending_route is set, forces the route even if the LLM deviates.
        """
        context_str = self._build_context_str(session)
        user_prompt = f"{context_str}\n\nUser message: {user_message}"

        print(f"[coordinator] Routing: \"{user_message}\"")
        print(f"[coordinator] Session context: {session.summary()}")
        if session.pending_route:
            print(f"[coordinator] Pending route: {session.pending_route}")

        try:
            response = self.mistral.chat.complete(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[coordinator] ⚠️  JSON invalide : {e} — raw={raw!r}")
            parsed = {"route": session.pending_route or "rag_query", "question": user_message, "missing": []}
        except Exception as e:
            print(f"[coordinator] ⚠️  Erreur API : {e}")
            parsed = {"route": session.pending_route or "rag_query", "question": user_message, "missing": []}

        print(f"[coordinator] Raw result: {parsed}")

        # ── Correction: force route if LLM deviates from pending_route ────────
        if session.pending_route and parsed.get("route") != session.pending_route:
            print(f"[coordinator] ⚡ LLM deviated to '{parsed.get('route')}' "
                  f"→ forced back to '{session.pending_route}' (pending_route)")
            parsed["route"] = session.pending_route

        # ── Fallback regex route ──────────────────────────────────────────
        # If the LLM returned rag_query but the message clearly expresses
        # a segmentation or frame intent, override the route.
        # This happens when the LLM "gives up" because required params are missing.
        if parsed.get("route") == "rag_query" and not session.pending_route:
            _seg_re = re.compile(
                r"\b(segment|segmentation|run segmentation|launch segmentation|"
                r"analyze|automatic analysis|detect lesion|use.{0,10}algorithm|"
                r"lancer|segmenter)\b",
                re.IGNORECASE,
            )
            _frame_re = re.compile(
                r"\b(show|display|view|frame|slice|render|visualize|"
                r"afficher|montrer)\b",
                re.IGNORECASE,
            )
            if _seg_re.search(user_message):
                parsed["route"] = "segmentation"
                print(f"[coordinator] 🔧 Route overridden rag_query → segmentation (keyword match)")
            elif _frame_re.search(user_message):
                parsed["route"] = "frame"
                print(f"[coordinator] 🔧 Route overridden rag_query → frame (keyword match)")

        # ── Fallback regex liaison_id ────────────────────────────────────
        # If the LLM failed to extract liaison_id but we're on the frame route,
        # try to extract the first integer directly from the user message.
        # Handles: "2", "lesion 2", "lesion number 2", "the 2nd lesion", "3rd structure"
        if parsed.get("route") == "frame" and parsed.get("liaison_id") is None:
            # Priority 1: ordinal form — "2nd", "3rd", "1st", "4th", etc.
            ord_match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", user_message, re.IGNORECASE)
            # Priority 2: bare integer
            num_match = re.search(r"\b(\d+)\b", user_message)
            match = ord_match or num_match
            if match:
                parsed["liaison_id"] = int(match.group(1))
                print(f"[coordinator] 🔧 liaison_id extracted via regex fallback: {parsed['liaison_id']}")


        # Update session with newly extracted parameters
        session.update(parsed)

        # Re-compute missing fields AFTER merging with session
        route = parsed.get("route", "rag_query")
        missing = self._compute_missing(route, session, parsed)

        # pending_route management:
        # - If fields are still missing → store the pending route
        # - If nothing is missing → clear the pending route
        if missing:
            session.pending_route = route
            # For rag_query: save the original question so it isn't lost
            # when the user replies with only "patient <id>" next turn.
            if route == "rag_query" and parsed.get("question"):
                session.pending_question = parsed["question"]
        else:
            session.pending_route = None
            # Restore pending_question if we're executing a rag_query
            # that was waiting for clarification.
            if route == "rag_query" and session.pending_question:
                parsed["question"] = session.pending_question
            session.pending_question = None

        result = {
            "route": route,
            "patient_id": session.patient_id,
            "accession_number": session.accession_number,
            "liaison_id": session.liaison_id,
            "view_type": session.view_type,
            "question": parsed.get("question") or user_message,
            "missing": missing,
            "clarification": self._build_clarification(missing) if missing else None,
        }

        print(f"[coordinator] → route={route} | missing={missing} | session={session.summary()}")
        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_context_str(session: SessionContext) -> str:
        """Generates the textual context description for the LLM."""
        lines = ["Current session context (already known, do not add to missing):"]
        lines.append(f"  patient_id: {session.patient_id or 'unknown'}")
        lines.append(f"  accession_number: {session.accession_number or 'unknown'}")
        lines.append(f"  liaison_id: {session.liaison_id if session.liaison_id is not None else 'unknown'}")
        lines.append(f"  view_type: {session.view_type if session.view_type is not None else 'unknown'}")
        if session.pending_route:
            lines.append(f"  pending_route: {session.pending_route}  "
                         "← IMPORTANT: user is answering a clarification request for this route. "
                         "You MUST set route to this value.")
        else:
            lines.append("  pending_route: none")
        return "\n".join(lines)

    @staticmethod
    def _compute_missing(route: str, session: SessionContext, parsed: dict) -> list[str]:
        """
        Checks required fields AFTER merging with session.
        Returns the list of still-missing fields.
        """
        required = {
            "segmentation": ["patient_id", "accession_number"],
            "frame": ["patient_id", "accession_number", "liaison_id", "view_type"],
            "rag_query": ["patient_id"],
        }
        needed = required.get(route, [])
        missing = []
        for field in needed:
            val = getattr(session, field, None)
            if val is None:
                missing.append(field)
        return missing

    @staticmethod
    def _build_clarification(missing: list[str]) -> str:
        """Generates a clarification request message for missing fields."""
        labels = {
            "patient_id": "the patient identifier (Patient ID)",
            "accession_number": "the accession number",
            "liaison_id": "the lesion/structure number",
            "view_type": "the view type (0 for 2D, 1 for 3D)",
        }
        items = [f"  • {labels.get(f, f)}" for f in missing]
        return "To continue, please provide the following:\n" + "\n".join(items)
