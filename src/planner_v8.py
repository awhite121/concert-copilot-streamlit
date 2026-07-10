from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
from collections import Counter
from .llm_client import generate_llm_response, LLMResult
from .places_client import search_nearby_plan_places

GROUNDING_RULES = """
You are Encore AI, an AI planner for live music.
Rules:
- Only recommend events in the provided event list.
- Only recommend nearby places in the provided nearby_places list.
- Do not invent concerts, venues, restaurants, prices, parking costs, or exact travel times.
- If prices, times, or nearby places are missing, say they are unknown or require verification.
- Be natural, specific, and helpful. Avoid robotic/static wording.
"""

def summarize_taste(top_artists, top_tracks):
    genres=[]
    for a in top_artists:
        genres.extend(a.get('genres', []) or [])
    return {"top_artist_names":[a.get('artist') for a in top_artists if a.get('artist')], "top_track_names":[t.get('track') for t in top_tracks if t.get('track')], "top_genres":[g for g,_ in Counter(genres).most_common(12)]}

def compact_event(e):
    return {k:e.get(k) for k in ["event_id","event_name","date","time","venue","city","state","genre","subgenre","artists","min_price","max_price","price_source","source","sources","all_urls","url","final_score","model_score","hybrid_score","embedding_score","exact_artist_score","genre_score","why_recommended"]}

def generate_taste_summary(top_artists, top_tracks):
    taste = summarize_taste(top_artists, top_tracks)
    prompt = f"""Create a concise music taste summary for a concert app.\nTaste data:\n{json.dumps(taste, indent=2)}\nReturn a paragraph plus 5 taste signals."""
    return generate_llm_response("taste_summary", GROUNDING_RULES, prompt, taste)

def explain_event(event, top_artists, top_tracks):
    taste = summarize_taste(top_artists, top_tracks)
    payload = compact_event(event)
    prompt = f"""Explain why this event fits the user in a natural, non-generic way.\nUser taste:\n{json.dumps(taste, indent=2)}\nEvent:\n{json.dumps(payload, indent=2)}\nMention price/source uncertainty when relevant. Return 2 short paragraphs and 3 bullets."""
    return generate_llm_response("event_explanation", GROUNDING_RULES, prompt, {"event":payload, **taste})

def compare_events(events, top_artists, top_tracks):
    taste = summarize_taste(top_artists, top_tracks)
    payload = [compact_event(e) for e in events[:5]]
    prompt = f"""Compare these retrieved concerts. Pick best overall, best value if price exists, and best discovery.\nUser taste:\n{json.dumps(taste, indent=2)}\nEvents:\n{json.dumps(payload, indent=2)}"""
    return generate_llm_response("compare_events", GROUNDING_RULES, prompt, {"events":payload, **taste})

def create_night_plan(event, top_artists, top_tracks, vibe, budget, notes):
    taste = summarize_taste(top_artists, top_tracks)
    payload = compact_event(event)
    places = search_nearby_plan_places(event.get('venue') or '', event.get('city') or '', event.get('state') or '', event.get('latitude'), event.get('longitude'), "restaurants bars coffee", 6)
    prompt = f"""Create a practical night-out plan around the selected concert.\nUser taste:\n{json.dumps(taste, indent=2)}\nSelected event:\n{json.dumps(payload, indent=2)}\nNearby places from Google Places, if configured:\n{json.dumps(places, indent=2)}\nPreferences: vibe={vibe}; budget={budget}; notes={notes}\nReturn: why it fits, nearby pre/post options, timing plan, budget watchouts, and what to verify before buying. Do not invent places or prices."""
    return generate_llm_response("night_plan", GROUNDING_RULES, prompt, {"event":payload, "nearby_places":places, **taste})

def filter_events_for_prompt(events, max_budget=None, weekend_only=False, keyword=None, top_n=8):
    out = list(events)
    if max_budget is not None:
        out = [e for e in out if e.get('min_price') is None or float(e.get('min_price')) <= max_budget]
    if weekend_only:
        out = [e for e in out if int(e.get('weekend_event') or 0) == 1]
    if keyword:
        kw=keyword.lower()
        out=[e for e in out if kw in str(e.get('event_name','')).lower() or kw in str(e.get('genre','')).lower() or kw in ' '.join(e.get('artists') or []).lower() or kw in str(e.get('why_recommended','')).lower()]
    return out[:top_n]

def planner_chat(question, ranked_events, top_artists, top_tracks, max_budget=None, weekend_only=False, keyword=None):
    taste=summarize_taste(top_artists, top_tracks)
    selected=filter_events_for_prompt(ranked_events, max_budget, weekend_only, keyword, 8)
    payload=[compact_event(e) for e in selected]
    prompt=f"""Answer the user's planning question using only these retrieved/ranked events.\nQuestion: {question}\nFilters: max_budget={max_budget}, weekend_only={weekend_only}, keyword={keyword}\nUser taste:\n{json.dumps(taste, indent=2)}\nEvents:\n{json.dumps(payload, indent=2)}\nReturn top 3 picks, why each fits, price caveats, and next action."""
    return generate_llm_response("planner_chat", GROUNDING_RULES, prompt, {"events":payload, **taste})
