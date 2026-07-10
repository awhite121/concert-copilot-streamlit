from __future__ import annotations
import requests
import urllib.parse
import os

from typing import List, Dict, Any
import json
import re
from collections import Counter
from datetime import datetime, timedelta, date

from .llm_client import generate_llm_response, LLMResult
from .places_client import search_nearby_plan_places

def summarize_taste(top_artists: List[Dict[str, Any]], top_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    genres = []
    for artist in top_artists:
        genres.extend(artist.get("genres", []) or [])
    top_genres = [g for g, _ in Counter(genres).most_common(12)]
    top_artist_names = [a.get("artist") for a in top_artists if a.get("artist")]
    top_track_names = [t.get("track") for t in top_tracks if t.get("track")]
    return {"top_artist_names": top_artist_names, "top_track_names": top_track_names, "top_genres": top_genres}

def compact_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "date": event.get("date"),
        "time": event.get("time"),
        "venue": event.get("venue"),
        "city": event.get("city"),
        "state": event.get("state"),
        "genre": event.get("genre"),
        "subgenre": event.get("subgenre"),
        "artists": event.get("artists", []),
        "min_price": event.get("min_price"),
        "max_price": event.get("max_price"),
        "median_price": event.get("median_price"),
        "average_price": event.get("average_price"),
        "listing_count": event.get("listing_count"),
        "price_source": event.get("price_source"),
        "source": event.get("source"),
        "sources": event.get("sources"),
        "url": event.get("url"),
        "all_urls": event.get("all_urls"),
        "final_score": event.get("final_score"),
        "model_score": event.get("model_score"),
        "hybrid_score": event.get("hybrid_score"),
        "embedding_rank_score": event.get("embedding_rank_score"),
        "exact_artist_score": event.get("exact_artist_score"),
        "genre_score": event.get("genre_score"),
        "why_recommended": event.get("why_recommended"),
    }

def filter_events_for_prompt(events, max_budget=None, weekend_only=False, keyword=None, top_n=8):
    filtered = events[:]
    if max_budget is not None:
        filtered = [e for e in filtered if e.get("min_price") is None or float(e.get("min_price")) <= max_budget]
    if weekend_only:
        filtered = [e for e in filtered if int(e.get("weekend_event") or 0) == 1]
    if keyword:
        kw = keyword.lower()
        filtered = [
            e for e in filtered
            if kw in str(e.get("event_name", "")).lower()
            or kw in str(e.get("genre", "")).lower()
            or kw in str(e.get("subgenre", "")).lower()
            or kw in " ".join(e.get("artists", []) or []).lower()
            or kw in str(e.get("why_recommended", "")).lower()
        ]
    return filtered[:top_n]

GROUNDING_RULES = """
You are Encore AI, an AI planner for live music.
Critical rules:
- Only recommend or mention events that appear in the provided event list.
- Only mention nearby places from the provided nearby_places list.
- Do not invent concerts, venues, ticket prices, artists, restaurants, bars, parking prices, or exact travel times.
- If price, time, or location details are missing, say they are missing/unknown.
- Use the recommender ranking, user taste, event metadata, and verified Google/Yelp place results as evidence.
- Never expose raw model jargon unless the user explicitly asks for technical details.
- Lead with a decisive recommendation, then explain it in plain English.
- Keep answers skimmable, practical, and product-like.
- The ranking model chooses candidates; you explain tradeoffs and turn the choice into an actionable plan.
"""

def generate_taste_summary(top_artists, top_tracks) -> LLMResult:
    taste = summarize_taste(top_artists, top_tracks)
    user_prompt = f"""
Create a concise music taste summary for a concert recommendation app.

Taste data:
{json.dumps(taste, indent=2)}

Return:
1. One-paragraph taste summary
2. 4-6 bullet signals the recommender should use
3. One sentence explaining how this powers concert recommendations
"""
    return generate_llm_response("taste_summary", GROUNDING_RULES, user_prompt, taste)

def explain_event(event, top_artists, top_tracks) -> LLMResult:
    top_artists = top_artists or []
    top_tracks = top_tracks or []
    vibe = night_style or vibe or kwargs.get("style") or "date night"
    notes = notes or kwargs.get("user_notes") or ""
    taste = summarize_taste(top_artists, top_tracks)
    event_payload = compact_event(event)
    user_prompt = f"""
Explain why this event was recommended in a natural, non-static tone.

User taste:
{json.dumps(taste, indent=2)}

Recommended event:
{json.dumps(event_payload, indent=2)}

Return:
- 2-3 sentence explanation that sounds like a real music discovery product
- 3 short bullets for strongest match signals
- 1 caveat if price/time/details are missing
Do not invent information beyond the supplied payload.
"""
    return generate_llm_response("event_explanation", GROUNDING_RULES, user_prompt, {"event": event_payload, **taste})

def compare_events(events, top_artists, top_tracks) -> LLMResult:
    taste = summarize_taste(top_artists, top_tracks)
    event_payload = [compact_event(e) for e in events[:5]]
    user_prompt = f"""
Compare these retrieved concerts for this user.

User taste:
{json.dumps(taste, indent=2)}

Events:
{json.dumps(event_payload, indent=2)}

Return exactly these sections:
## Best overall
One decisive pick and a two-sentence reason.
## Best value
Use only published/returned prices. If no reliable price exists, say so.
## Best discovery
One adventurous pick tied to the user's actual taste.
## Quick comparison
A compact markdown table with Show, Date, Venue, Price, and Best for.
Do not mention any event outside the list.
"""
    return generate_llm_response("compare_events", GROUNDING_RULES, user_prompt, {"events": event_payload, **taste})

def create_night_plan(event, top_artists=None, top_tracks=None, vibe=None, budget=None, notes="", nearby_places=None, night_style=None, **kwargs) -> LLMResult:
    taste = summarize_taste(top_artists, top_tracks)
    event_payload = compact_event(event)
    if nearby_places is None:
        nearby_places = search_nearby_plan_places(
            venue_name=event.get("venue") or "",
            city=event.get("city") or "",
            state=event.get("state") or "",
            latitude=event.get("latitude"),
            longitude=event.get("longitude"),
            place_kind="restaurants bars coffee",
            max_results=10,
            radius_miles=1.5,
            night_style=vibe,
        )
    user_prompt = f"""
Create a practical concert night plan around this selected event.

User taste:
{json.dumps(taste, indent=2)}

Selected event:
{json.dumps(event_payload, indent=2)}

Verified nearby places from Google Places and/or Yelp, if available:
{json.dumps(nearby_places, indent=2)}

User preferences:
- Vibe: {vibe}
- Night style: {vibe}
- Notes: {notes}

Return a clean, concise plan with these sections:
## The plan
A short summary of the night and why the concert fits.
## Before the show
Choose verified nearby places that best match the requested night style and explain why they fit.
## Timing
Use the supplied event time. Any other timing must be clearly described as an estimate.
## Verify before you go
List only the important unknowns such as doors, hours, transportation, or missing prices.
Do not invent restaurants, bars, parking prices, ticket prices, or exact travel times.
If nearby_places is empty, say no verified nearby options were returned.
"""
    return generate_llm_response(
        "night_plan",
        GROUNDING_RULES,
        user_prompt,
        {"event": event_payload, "nearby_places": nearby_places, "vibe": vibe, "budget": budget, **taste},
    )


def _filter_by_request_window(events, question):
    """
    Parse common date windows from Copilot prompts.
    The app can only filter events already retrieved by the sidebar search.

    Supported examples:
    - next 2 months, next 8 months, next 30 days, next 6 weeks
    - this month, next month
    - September 1 - October 1, Sep 1 to Oct 1, Sept 1 through Oct 1
    - 9/1 - 10/1 or 09/01/2026 to 10/01/2026
    """
    q = (question or "").lower()
    today = date.today()
    start = today
    end = None
    label = None

    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    def _resolve_year(month: int, explicit_year=None):
        if explicit_year:
            return int(explicit_year)
        year = today.year
        # If a user asks for a past month while today's month is late in the year,
        # assume they mean the next calendar occurrence.
        if month < today.month and today.month >= 10:
            year += 1
        return year

    # Named month range: September 1 - October 1, Sep 1 to Oct 1, etc.
    named = re.search(
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:,?\s*(\d{4}))?\s*(?:-|to|through|until|and)\s*"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:,?\s*(\d{4}))?",
        q,
    )
    if named:
        m1, d1, y1, m2, d2, y2 = named.groups()
        mo1, mo2 = month_map[m1], month_map[m2]
        start = date(_resolve_year(mo1, y1), mo1, int(d1))
        end = date(_resolve_year(mo2, y2 or y1), mo2, int(d2))
        if end < start:
            end = date(end.year + 1, end.month, end.day)
        label = f"{start.strftime('%b %-d')} to {end.strftime('%b %-d')}"
    else:
        numeric = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\s*(?:-|to|through|until|and)\s*(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", q)
        if numeric:
            m1, d1, y1, m2, d2, y2 = numeric.groups()
            def norm_year(y, month):
                if not y:
                    return _resolve_year(int(month))
                y = int(y)
                return 2000 + y if y < 100 else y
            start = date(norm_year(y1, m1), int(m1), int(d1))
            end = date(norm_year(y2 or y1, m2), int(m2), int(d2))
            if end < start:
                end = date(end.year + 1, end.month, end.day)
            label = f"{start.strftime('%b %-d')} to {end.strftime('%b %-d')}"

    if end is None:
        match = re.search(r"next\s+(\d+)\s*(day|days|week|weeks|month|months)", q)
        if match:
            qty = int(match.group(1))
            unit = match.group(2)
            if unit.startswith("day"):
                end = start + timedelta(days=qty)
                label = f"next {qty} days"
            elif unit.startswith("week"):
                end = start + timedelta(weeks=qty)
                label = f"next {qty} weeks"
            elif unit.startswith("month"):
                end = start + timedelta(days=qty * 31)
                label = f"next {qty} months"
        elif "this month" in q:
            end = start.replace(day=28) + timedelta(days=4)
            end = end.replace(day=1)
            label = "this month"
        elif "next month" in q:
            first_next = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
            start = first_next
            end = (first_next.replace(day=28) + timedelta(days=4)).replace(day=1)
            label = "next month"

    if not end:
        return events, None

    filtered = []
    for event in events or []:
        dt = _parse_event_datetime(event)
        if not dt:
            continue
        d = dt.date()
        if start <= d <= end:
            filtered.append(event)

    requested_window = {
        "label": label,
        "start": str(start),
        "end": str(end),
        "matched_count": len(filtered),
    }
    return filtered, requested_window


def planner_chat(question, ranked_events, top_artists, top_tracks, max_budget=None, weekend_only=False, keyword=None, search_start_date=None, search_end_date=None) -> LLMResult:
    taste = summarize_taste(top_artists, top_tracks)
    windowed_events, requested_window = _filter_by_request_window(ranked_events, question)
    events_for_filter = windowed_events if requested_window and windowed_events else ranked_events
    selected_events = filter_events_for_prompt(events_for_filter, max_budget=max_budget, weekend_only=weekend_only, keyword=keyword, top_n=10)
    event_payload = [compact_event(e) for e in selected_events]
    request_note = "No explicit date window detected in the typed request."
    if requested_window:
        request_note = f"Typed request asked for {requested_window['label']} ({requested_window['start']} to {requested_window['end']}); {requested_window['matched_count']} retrieved events matched that window."
    available_range = {"search_start_date": str(search_start_date), "search_end_date": str(search_end_date)}
    user_prompt = f"""
Answer the user's concert planning question using only the retrieved event list.

User question:
{question}

Date/window interpretation:
- {request_note}
- Current sidebar search range: {json.dumps(available_range)}
- If the typed request asks for a longer window than the current retrieved search range, say the current results only cover the sidebar date range and tell the user to expand the sidebar dates.

Filters applied:
- Max budget: {max_budget}
- Weekend only: {weekend_only}
- Keyword: {keyword}

User taste:
{json.dumps(taste, indent=2)}

Retrieved/ranked events after request filtering:
{json.dumps(event_payload, indent=2)}

Return exactly these sections:
## My pick
Answer the question directly and choose one best option when possible. Respect the typed date window when possible.
## Other strong options
Up to two alternatives, each with one plain-English tradeoff.
## What to know
Call out missing prices, unknown times, current sidebar date limits, or other uncertainty without overexplaining.
## Next move
One clear action the user should take in the app.
"""
    return generate_llm_response("planner_chat", GROUNDING_RULES, user_prompt, {"events": event_payload, "requested_window": requested_window, "available_range": available_range, **taste})



def _parse_event_datetime(event):
    event = event or {}
    date_value = event.get("date") or event.get("event_date") or event.get("localDate") or event.get("start_date")
    time_value = event.get("time") or event.get("event_time") or event.get("localTime") or event.get("start_time") or "20:00:00"
    dt_value = event.get("event_datetime") or event.get("datetime") or event.get("datetime_local")
    if dt_value and "T" in str(dt_value):
        try:
            return datetime.fromisoformat(str(dt_value).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    if not date_value:
        return None
    try:
        return datetime.strptime(f"{str(date_value)[:10]} {str(time_value)[:8]}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.strptime(f"{str(date_value)[:10]} {str(time_value)[:5]}", "%Y-%m-%d %H:%M")
        except Exception:
            return None


def _clock(value):
    if value is None:
        return "Time TBD"
    return value.strftime("%-I:%M %p")


def _place_category(place):
    text = " ".join([str(place.get("type") or ""), str(place.get("name") or "")]).lower()
    if any(word in text for word in ["restaurant", "food", "taco", "pizza", "dining", "kitchen", "grill", "cafe"]):
        return "food"
    if any(word in text for word in ["bar", "pub", "cocktail", "brew", "wine", "lounge"]):
        return "drink"
    if any(word in text for word in ["coffee", "tea", "bakery"]):
        return "coffee"
    return "other"




def _normalize_place_list_for_plan(nearby_places):
    """Flatten and keep only dict-like place objects so sorting does not crash."""
    output = []
    def add(item):
        if isinstance(item, dict):
            output.append(item)
        elif isinstance(item, (list, tuple)):
            for child in item:
                add(child)
    add(nearby_places or [])
    return output

def build_structured_night_plan(event, night_style="date night", notes="", radius_miles=1.5, place_focus="auto"):
    """Structured planner for the UI. Uses event metadata + real nearby places when keys exist.
    Never returns blank timeline cells; every card has a real place/detail.
    """
    event = event or {}

    def pick(*keys, default=""):
        for k in keys:
            v = event.get(k) if isinstance(event, dict) else None
            if v not in (None, "", "None", "nan", "Date TBD", "Time TBD"):
                return str(v)
        return default

    title = pick("event_name", "title", "name", default="Selected show")
    venue = pick("venue", "venue_name", "location_name", default="the venue")
    city = pick("city", "event_city", "venue_city", default="the area")
    state = pick("state", "region", "venue_state", default="")
    date_text = pick("date", "event_date", "localDate", "start_date", default="show day")
    time_text = pick("time", "event_time", "localTime", "start_time", default="show time")
    genre = pick("winning_genre_cluster_label", "genre", "subgenre", default="music")
    score = event.get("final_score") or event.get("model_score") or event.get("hybrid_score")
    try:
        match_score = max(0, min(100, float(score))) if score is not None else None
    except Exception:
        match_score = None

    lat = event.get("latitude") or event.get("lat")
    lon = event.get("longitude") or event.get("lon")

    place_kind = "restaurants bars coffee"
    low = f"{night_style} {notes} {place_focus}".lower()
    if "late" in low or "after" in low or "party" in low:
        place_kind = "cocktail bars late night food nightlife"
    elif "date" in low or "dinner" in low:
        place_kind = "restaurants cocktail bars wine bars"
    elif "chill" in low or "coffee" in low:
        place_kind = "coffee dessert wine bars relaxed restaurants"
    elif "high" in low or "pregame" in low or "group" in low:
        place_kind = "bars breweries casual restaurants"

    places = []
    try:
        places = search_nearby_plan_places(
            venue_name=venue,
            city=city,
            state=state,
            latitude=lat,
            longitude=lon,
            place_kind=place_kind,
            max_results=12,
            radius_miles=radius_miles or 1.5,
            night_style=night_style or "date night",
        ) or []
    except Exception:
        places = []

    # Stable fallback so the UI never shows empty cards.
    if not places:
        places = [
            {"name": f"Dinner or drinks near {venue}", "category": "Pre-show option", "type": "Pre-show option", "rating": None, "distance_miles": None, "address": city, "url": ""},
            {"name": f"Backup spot near {venue}", "category": "Backup option", "type": "Backup option", "rating": None, "distance_miles": None, "address": city, "url": ""},
            {"name": "Easy nearby after-show option", "category": "After-show option", "type": "After-show option", "rating": None, "distance_miles": None, "address": city, "url": ""},
        ]

    def is_bar(place):
        txt = f"{place.get('name','')} {place.get('type','')} {place.get('category','')}".lower()
        return any(w in txt for w in ["bar", "cocktail", "lounge", "wine", "brew", "nightlife"])

    def is_food(place):
        txt = f"{place.get('name','')} {place.get('type','')} {place.get('category','')}".lower()
        return any(w in txt for w in ["restaurant", "food", "kitchen", "grill", "taco", "cafe", "dinner"])

    pre = next((p for p in places if is_food(p)), places[0])
    after = next((p for p in places if is_bar(p) and p.get("name") != pre.get("name")), places[1] if len(places) > 1 else places[0])
    backup = next((p for p in places if p.get("name") not in {pre.get("name"), after.get("name")}), places[2] if len(places) > 2 else after)

    dt = _parse_event_datetime({"date": date_text, "time": time_text, **event})
    if dt:
        pre_time = _clock(dt - timedelta(minutes=90))
        head_time = _clock(dt - timedelta(minutes=45))
        show_time = _clock(dt)
        after_time = _clock(dt + timedelta(hours=2, minutes=15))
        date_label = dt.strftime("%a, %b %-d") if os.name != "nt" else dt.strftime("%a, %b %#d")
    else:
        pre_time = "2 hours before"
        head_time = "45 minutes before"
        show_time = time_text or "Show time"
        after_time = "After show"
        date_label = date_text

    def place_line(place):
        bits = []
        if place.get("rating"):
            bits.append(f"{place.get('rating')}★")
        if place.get("distance_miles") is not None:
            bits.append(f"{place.get('distance_miles')} mi")
        if place.get("address"):
            bits.append(str(place.get("address")))
        return " · ".join(bits)

    timeline = [
        {"time": pre_time, "title": "Pre-show", "place": pre.get("name"), "description": f"Start with {pre.get('name')} so the night has a real first stop. {place_line(pre)}"},
        {"time": head_time, "title": "Head to venue", "place": venue, "description": "Leave a buffer for parking, rideshare, doors, security, and finding your seat."},
        {"time": show_time, "title": "Show", "place": title, "description": f"{genre} fit" + (f" · Match Score {match_score:.0f}" if match_score is not None else "")},
        {"time": after_time, "title": "After show", "place": after.get("name"), "description": f"Use {after.get('name')} if you want to keep the night going. Backup: {backup.get('name')}."},
    ]

    pre_details = place_line(pre) or "close to the venue"
    after_details = place_line(after) or "nearby option"
    backup_details = place_line(backup) or "backup nearby"
    full_plan = f"""
### The move
Go with **{title}** at **{venue}** in **{city}** on **{date_label}** at **{show_time}**. For a **{night_style}** night, start at **{pre.get('name')}**, keep the venue buffer simple, then use **{after.get('name')}** as the after-show move.

### Timeline
- **{pre_time}** — **{pre.get('name')}** before the show. {pre_details}
- **{head_time}** — Head to **{venue}** with time for parking, rideshare, doors, and security.
- **{show_time}** — **{title}**. {genre} fit{f' with Match Score {match_score:.0f}' if match_score is not None else ''}.
- **{after_time}** — **{after.get('name')}** if you want to keep going. {after_details}

### Backup option
Use **{backup.get('name')}** if the first spot is packed, closed, or not the vibe. {backup_details}

### Verify before you go
Check ticket availability, door time, restaurant/bar hours, reservations, transportation, parking, and bag policy for **{venue}**.
"""

    return {
        "event": event,
        "title": title,
        "venue": venue,
        "city": city,
        "state": state,
        "date": date_text,
        "time": time_text,
        "timeline": timeline,
        "nearby_options": places,
        "places": places,
        "summary": full_plan,
        "full_plan": full_plan,
        "primary_place": pre,
        "after_place": after,
        "backup_place": backup,
    }
