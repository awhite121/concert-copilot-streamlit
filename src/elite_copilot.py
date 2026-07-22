from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import math

from .llm_client import generate_llm_response, LLMResult
from .planner import summarize_taste, compact_event, GROUNDING_RULES
from .places_client import search_nearby_plan_places
from .weather_client import get_event_weather


def _num(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _known_price(event: Dict[str, Any]) -> Optional[float]:
    for field in ["min_price", "median_price", "average_price", "max_price"]:
        value = event.get(field)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _price_label(event: Dict[str, Any]) -> str:
    price = _known_price(event)
    if price is None:
        return "Live price not returned"
    if event.get("max_price") and isinstance(event.get("max_price"), (int, float)) and float(event.get("max_price")) > price:
        return f"${price:.0f}+ / up to ${float(event.get('max_price')):.0f}"
    return f"From about ${price:.0f}"


def _date_label(event: Dict[str, Any]) -> str:
    date = str(event.get("date") or "")[:10]
    time = str(event.get("time") or "")[:5]
    if not date:
        return "Date TBD"
    try:
        dt = datetime.strptime(f"{date} {time or '20:00'}", "%Y-%m-%d %H:%M")
        return dt.strftime("%a, %b %-d · %-I:%M %p")
    except Exception:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            return dt.strftime("%a, %b %-d")
        except Exception:
            return date


def _weather_penalty(weather: Dict[str, Any]) -> float:
    if not weather or not weather.get("available"):
        return 0.0
    penalty = 0.0
    pop = weather.get("precip_probability")
    temp = weather.get("temperature_f")
    wind = weather.get("wind_mph")
    if isinstance(pop, (int, float)):
        penalty += max(0.0, min(12.0, pop * 16.0))
    if isinstance(temp, (int, float)) and (temp < 45 or temp > 92):
        penalty += 4.0
    if isinstance(wind, (int, float)) and wind > 18:
        penalty += 3.0
    return penalty


def _place_strength(places: List[Dict[str, Any]]) -> float:
    if not places:
        return 0.0
    best = places[:5]
    score = 0.0
    for p in best:
        rating = _num(p.get("rating"), 0.0)
        distance = _num(p.get("distance_miles"), 1.5)
        score += max(0.0, (rating - 3.3) * 12.0) + max(0.0, 1.2 - distance) * 10.0
    return min(100.0, score / max(1, len(best)) * 2.2)




def _event_hour(event: Dict[str, Any]) -> Optional[int]:
    time_value = str(event.get("time") or "").strip()
    if not time_value:
        return None
    try:
        return int(time_value[:2])
    except Exception:
        return None


def _late_place_strength(places: List[Dict[str, Any]]) -> float:
    if not places:
        return 0.0
    score = 0.0
    for p in places[:6]:
        text = f"{p.get('name','')} {p.get('type','')}".lower()
        rating = _num(p.get("rating"), 0.0)
        distance = _num(p.get("distance_miles"), 1.5)
        late_hit = any(word in text for word in ["bar", "cocktail", "wine", "lounge", "night", "music", "club", "brew", "pub"])
        open_bonus = 4 if p.get("open_now") is True else 0
        score += max(0.0, (rating - 3.4) * 10.0) + max(0.0, 1.5 - distance) * 6.0 + (8 if late_hit else 0) + open_bonus
    return min(100.0, score / max(1, min(6, len(places))) * 1.9)

def _core_context(event: Dict[str, Any], places: Optional[List[Dict[str, Any]]] = None, weather: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    places = places or []
    weather = weather or {}
    price = _known_price(event)
    base_score = _num(event.get("final_score") or event.get("hybrid_score"), 0.0)
    direct = _num(event.get("has_direct_artist_match"), 0.0)
    genre = _num(event.get("genre_cluster_score") or event.get("genre_score"), 0.0)
    venue = _num(event.get("venue_quality_signal"), 0.0)
    discovery = _num(event.get("discovery_quality_score"), 0.0)
    weekend = _num(event.get("weekend_event"), 0.0)
    source_count = _num(event.get("source_count") or len(event.get("sources") or []), 1.0)
    price_score = _num(event.get("price_score"), 35.0)
    place_score = _place_strength(places)
    weather_risk = _weather_penalty(weather)
    confidence = min(100.0, 24.0 + base_score * 0.32 + direct * 16 + min(source_count, 3) * 8 + (8 if price is not None else 0) + (7 if event.get("image_url") else 0))
    logistics = min(100.0, venue * 0.24 + price_score * 0.22 + place_score * 0.28 + weekend * 8 + (8 if event.get("known_event_time") else 0) + min(source_count, 3) * 5 - weather_risk)
    date_night = min(100.0, base_score * 0.32 + place_score * 0.33 + venue * 0.18 + price_score * 0.10 + weekend * 5 - weather_risk)
    value = min(100.0, base_score * 0.42 + price_score * 0.35 + min(source_count, 3) * 7 + (7 if price is not None else -4))
    low_effort = min(100.0, logistics * 0.65 + base_score * 0.25 + (10 if price is not None else 0))
    late_places = _late_place_strength(places)
    hour = _event_hour(event)
    late_event_bonus = 14 if hour is not None and hour >= 21 else 6 if hour is not None and hour >= 20 else 0
    late_night = min(100.0, base_score * 0.24 + late_places * 0.44 + venue * 0.12 + late_event_bonus + min(source_count, 3) * 4 - weather_risk)
    top_artist = min(100.0, base_score * 0.52 + direct * 28 + confidence * 0.18 + min(source_count, 3) * 4)
    compatible = min(100.0, base_score * 0.50 + confidence * 0.24 + genre * 0.18 + place_score * 0.08)
    night_out = min(100.0, base_score * 0.36 + place_score * 0.34 + logistics * 0.20 + confidence * 0.10)
    return {
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "artist_or_title": event.get("event_name"),
        "date_label": _date_label(event),
        "venue": event.get("venue"),
        "city": event.get("city"),
        "state": event.get("state"),
        "price_label": _price_label(event),
        "ticket_url": event.get("url"),
        "spotify_links": event.get("spotify_links"),
        "why_recommended": event.get("why_recommended"),
        "reason_tags": event.get("reason_tags"),
        "confidence_label": event.get("confidence_label") or event.get("match_confidence") or "Recommendation",
        "final_score": round(base_score, 2),
        "music_fit_score": round(base_score, 2),
        "direct_artist_match": bool(direct),
        "genre_cluster_score": round(genre, 2),
        "discovery_quality_score": round(discovery, 2),
        "venue_quality_signal": round(venue, 2),
        "source_count": source_count,
        "price_available": price is not None,
        "place_score": round(place_score, 2),
        "weather": weather,
        "nearby_places": places[:6],
        "copilot_scores": {
            "overall": round(min(100.0, base_score * 0.60 + logistics * 0.25 + confidence * 0.15), 2),
            "date_night": round(date_night, 2),
            "value": round(value, 2),
            "discovery": round(min(100.0, discovery * 0.52 + genre * 0.25 + base_score * 0.23), 2),
            "low_effort": round(low_effort, 2),
            "late_night": round(late_night, 2),
            "top_artist": round(top_artist, 2),
            "compatible": round(compatible, 2),
            "night_out": round(night_out, 2),
            "request_match": round(min(100.0, base_score * 0.60 + logistics * 0.25 + confidence * 0.15), 2),
            "confidence": round(confidence, 2),
            "logistics": round(logistics, 2),
        },
        "tradeoffs": _tradeoffs(event, places, weather),
    }


def _tradeoffs(event: Dict[str, Any], places: List[Dict[str, Any]], weather: Dict[str, Any]) -> List[str]:
    items = []
    if _known_price(event) is None:
        items.append("ticket price was not returned by connected sources")
    if not event.get("time"):
        items.append("show time is incomplete")
    if not places:
        items.append("nearby place context is missing or API did not return matches")
    if weather and weather.get("available") and weather.get("risk_summary") and "no major" not in str(weather.get("risk_summary")):
        items.append(f"weather flag: {weather.get('risk_summary')}")
    if _num(event.get("source_count") or len(event.get("sources") or []), 1.0) <= 1:
        items.append("single-source event listing")
    return items[:4]


def enrich_candidate_context(events: List[Dict[str, Any]], vibe: str, radius_miles: float = 0.75, enrich_top_n: int = 8, use_places: bool = True, use_weather: bool = False) -> List[Dict[str, Any]]:
    contexts = []
    for i, event in enumerate(events):
        places = []
        weather = {"available": False, "reason": "weather not requested"}
        if i < enrich_top_n and use_places:
            places = search_nearby_plan_places(
                venue_name=event.get("venue") or "",
                city=event.get("city") or "",
                state=event.get("state") or "",
                latitude=event.get("latitude"),
                longitude=event.get("longitude"),
                max_results=8,
                radius_miles=radius_miles,
                use_google=True,
                use_yelp=True,
                night_style=vibe,
            )
        if i < enrich_top_n and use_weather:
            weather = get_event_weather(event, event.get("city") or "", event.get("state") or "")
        contexts.append(_core_context(event, places, weather))
    return contexts



def _score_key_for_situation(situation: str) -> str:
    text = str(situation or "auto").lower()
    if any(word in text for word in ["value", "budget", "cheap", "affordable", "under $"]):
        return "value"
    if "artist" in text or "familiar" in text:
        return "top_artist"
    if "compatible" in text or "compat" in text:
        return "compatible"
    if "discover" in text or "new" in text:
        return "discovery"
    if "date" in text or "romantic" in text:
        return "date_night"
    if "late" in text or "after" in text:
        return "late_night"
    if "effort" in text or "easy" in text or "simple" in text:
        return "low_effort"
    if "night out" in text or "fun night" in text:
        return "night_out"
    return "overall"

def _pick_unique(contexts: List[Dict[str, Any]], key: str, used: set) -> Optional[Dict[str, Any]]:
    candidates = [c for c in contexts if c.get("event_id") not in used]
    if not candidates:
        candidates = contexts[:]
    if not candidates:
        return None
    chosen = sorted(candidates, key=lambda c: c.get("copilot_scores", {}).get(key, 0), reverse=True)[0]
    used.add(chosen.get("event_id"))
    return chosen



def select_copilot_picks(contexts: List[Dict[str, Any]], situation: str = "auto", user_goal: str = "") -> Dict[str, Dict[str, Any]]:
    used = set()
    intent = f"{situation or ''} {user_goal or ''}".strip()
    request_key = _score_key_for_situation(intent)
    for context in contexts:
        scores = context.setdefault("copilot_scores", {})
        scores["request_match"] = scores.get(request_key, scores.get("overall", 0))
    best_night_key = "late_night" if any(word in intent.lower() for word in ["late", "after"]) else "night_out"
    return {
        "Best choice": _pick_unique(contexts, "request_match", used),
        "Artist you know": _pick_unique(contexts, "top_artist", used),
        "Best discovery": _pick_unique(contexts, "discovery", used),
        "Best night out": _pick_unique(contexts, best_night_key, used),
    }

def build_context_brief(contexts: List[Dict[str, Any]], picks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "candidate_count": len(contexts),
        "data_points_used": [
            "Spotify direct artist match",
            "Spotify genre/taste cluster fit",
            "recommender final score",
            "trained model score when active",
            "ticket/event source count",
            "venue quality signal",
            "event time and late-night suitability",
            "nearby Google/Yelp place ratings, distance, type, and price level when configured",
            "OpenWeather context when configured",
            "feedback-trained ranking signals",
        ],
        "picks": {k: v for k, v in picks.items() if v},
        "top_candidates": contexts[:12],
    }


def generate_elite_copilot_report(question: str, contexts: List[Dict[str, Any]], picks: Dict[str, Dict[str, Any]], top_artists, top_tracks, situation: str, budget: Optional[float] = None) -> LLMResult:
    taste = summarize_taste(top_artists, top_tracks)
    brief = build_context_brief(contexts, picks)
    user_prompt = f"""
You are the elite version of Encore AI. Make a decision, not a generic summary.

User question/request:
{question}

User situation:
- Situation: {situation}
- Budget cap: ignored unless reliable event price exists; current app does not use sparse concert prices as the main filter

User taste:
{json.dumps(taste, indent=2)}

Candidate context and deterministic Copilot picks:
{json.dumps(brief, indent=2)}

Return exactly these sections:
## Copilot decision
Pick the single best move and explain why in 2-3 sentences.
## Picks by purpose
Cover Best choice, Artist you know, Best discovery, and Best night out. Include one honest tradeoff each.
## Why this is not just the ranking model
Explain how event fit, venue, timing, places, reviews/ratings, distance, place price level, and uncertainty changed the decision.
## What I would do next
Give one clear next action in the app.

Rules:
- Only mention events included in candidate context.
- Only mention places included in nearby_places.
- Do not invent prices, weather, venues, travel time, restaurants, bars, or artists.
- If place/weather/price data is missing, say it is missing.
"""
    return generate_llm_response(
        "elite_copilot",
        GROUNDING_RULES + "\nYou are decisive, specific, and grounded. You do not invent data.",
        user_prompt,
        {"events": contexts[:12], "picks": picks, "situation": situation, **taste},
    )


def fallback_pick_summary(picks: Dict[str, Dict[str, Any]]) -> str:
    lines = ["## Copilot decision"]
    best = picks.get("Best choice")
    if best:
        lines.append(f"The strongest move is **{best.get('event_name')}** because it has the best combined music fit, confidence, and night-out score among the retrieved options.")
    else:
        lines.append("No ranked events are available yet. Run a concert search first.")
    lines.append("\n## Picks by purpose")
    for label, item in picks.items():
        if not item:
            continue
        trade = "; ".join(item.get("tradeoffs") or []) or "no major caveat from returned data"
        lines.append(f"- **{label}:** {item.get('event_name')} — {item.get('date_label')} at {item.get('venue') or 'Venue TBD'}. Tradeoff: {trade}.")
    lines.append("\n## Why this is not just the ranking model")
    lines.append("The Copilot layer looks beyond pure music fit by using venue signals, source confidence, event timing, nearby-place quality, ratings, distance, place types, weather when configured, and missing-data caveats.")
    lines.append("\n## What I would do next")
    lines.append("Open the top ticket link, verify the price/time, then save the event to your shortlist if the logistics work.")
    return "\n".join(lines)
