from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import time

from .config import get_secret

@dataclass
class LLMResult:
    text: str
    used_llm: bool
    model: str
    latency_ms: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None

def get_openai_model() -> str:
    return get_secret("OPENAI_MODEL", "gpt-5.4-mini")

def _fallback_response(task: str, context: Dict[str, Any]) -> str:
    if task == "taste_summary":
        artists = ", ".join(context.get("top_artist_names", [])[:8])
        genres = ", ".join(context.get("top_genres", [])[:8])
        return (
            f"Your taste profile leans toward {genres or 'a mixed set of genres'}. "
            f"Your strongest artist signals are {artists or 'not available yet'}. "
            "The recommender uses those taste signals to score nearby concerts by artist match, genre overlap, and semantic similarity."
        )
    if task == "event_explanation":
        event = context.get("event", {})
        return (
            f"{event.get('event_name', 'This event')} is recommended because it scored highly on the app's ranking signals. "
            f"The strongest reason is: {event.get('why_recommended', 'similarity to your taste profile')}. "
            "This explanation is generated from retrieved event data only."
        )
    if task == "compare_events":
        events = context.get("events", [])
        lines = ["Here is a grounded comparison of the retrieved events:"]
        for i, e in enumerate(events[:5], start=1):
            lines.append(
                f"{i}. {e.get('event_name')} at {e.get('venue')} on {e.get('date')} — "
                f"score {e.get('final_score')}. Reason: {e.get('why_recommended')}"
            )
        return "\n".join(lines)
    if task == "night_plan":
        event = context.get("event", {})
        budget = context.get("budget", "not specified")
        vibe = context.get("vibe", "flexible")
        return (
            f"Plan idea for {event.get('event_name', 'the selected show')}:\n"
            f"- Show: {event.get('event_name')} at {event.get('venue')} on {event.get('date')}.\n"
            f"- Vibe: {vibe}.\n"
            f"- Budget: {budget}.\n"
            "- Arrive 30-45 minutes before start time, check ticket/parking details, and keep dinner nearby simple.\n"
            "- I am only using the event data returned by the app, not inventing venues or concerts."
        )
    events = context.get("events", [])
    lines = ["Based on the retrieved and ranked events, these are the strongest options:"]
    for i, e in enumerate(events[:5], start=1):
        lines.append(
            f"{i}. {e.get('event_name')} — {e.get('date')} at {e.get('venue')} "
            f"({e.get('city')}, {e.get('state')}), score {e.get('final_score')}. "
            f"Why: {e.get('why_recommended')}"
        )
    lines.append("I did not invent any events; these options came from the event API and recommender output.")
    return "\n".join(lines)

def generate_llm_response(task: str, system_prompt: str, user_prompt: str, context: Dict[str, Any]) -> LLMResult:
    api_key = get_secret("OPENAI_API_KEY")
    model = get_openai_model()
    started = time.time()
    if not api_key:
        return LLMResult(
            text=_fallback_response(task, context),
            used_llm=False,
            model="fallback-no-api-key",
            latency_ms=int((time.time() - started) * 1000),
            error="OPENAI_API_KEY not set",
        )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(response, "output_text", None) or str(response)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        return LLMResult(
            text=text,
            used_llm=True,
            model=model,
            latency_ms=int((time.time() - started) * 1000),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        return LLMResult(
            text=_fallback_response(task, context),
            used_llm=False,
            model=model,
            latency_ms=int((time.time() - started) * 1000),
            error=str(e),
        )
