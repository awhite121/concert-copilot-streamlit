import uuid
import time
import os
import math
import urllib.parse
import re
from html import escape
from datetime import date, datetime, timedelta
from typing import Dict, Any, List

import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import init_db, clear_all_feedback, insert_llm_call, load_llm_calls, load_user_latest_preferences
from src.spotify_client import (
    get_spotify_client,
    get_current_user,
    get_blended_taste_profile,
    demo_profile,
    add_group_listener_artists,
)
from src.event_sources import search_all_sources
from src.recommender import rank_events_v6
from src.feedback import (
    save_feedback_action,
    clear_feedback_preference,
    log_impressions_once,
    get_feedback_df,
    summarize_feedback,
    get_user_rated_event_ids,
    get_user_not_for_me_event_ids,
    get_user_shortlist_df,
    interaction_row_to_event,
)
from src.modeling import (
    train_feedback_model,
    load_feedback_model,
    rollback_to_previous,
    list_model_versions,
    compare_model_variants,
)
from src.planner import (
    generate_taste_summary,
    compare_events,
    create_night_plan,
    planner_chat,
    build_structured_night_plan,
    _filter_by_request_window,
)
from src.places_client import search_nearby_plan_places, test_google_places_key, test_yelp_key
from src.weather_client import test_openweather_key
from src.elite_copilot import (
    enrich_candidate_context,
    select_copilot_picks,
    generate_elite_copilot_report,
    fallback_pick_summary,
)
from src.event_grouping import collapse_ranked_events
from src.genre_clusters import build_user_taste_clusters, cluster_label


# === HOTFIX V34: display dedupe + saved-memory helpers ===
def _cc_clean_text(v):
    if v is None:
        return ""
    return str(v).strip()

def _cc_norm_text(v):
    return "".join(ch.lower() for ch in _cc_clean_text(v) if ch.isalnum())

def _cc_pick(event, keys):
    if not isinstance(event, dict):
        return ""
    for key in keys:
        val = event.get(key)
        if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
            return val
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, dict):
        for key in keys:
            val = raw.get(key)
            if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
                return val
        dates = raw.get("dates") or {}
        if isinstance(dates, dict):
            start = dates.get("start") or {}
            if isinstance(start, dict):
                joined = " ".join(keys).lower()
                if "date" in joined:
                    return start.get("localDate") or start.get("dateTime") or ""
                if "time" in joined:
                    return start.get("localTime") or start.get("dateTime") or ""
        embedded = raw.get("_embedded") or {}
        if isinstance(embedded, dict):
            venues = embedded.get("venues") or []
            if venues and isinstance(venues, list) and isinstance(venues[0], dict):
                joined = " ".join(keys).lower()
                if "venue" in joined:
                    return venues[0].get("name") or ""
                if "city" in joined:
                    city = venues[0].get("city") or {}
                    if isinstance(city, dict):
                        return city.get("name") or ""
                    return city or ""
    return ""

def _cc_event_title(event):
    if not isinstance(event, dict):
        return ""
    for k in ["title", "event_name", "name", "event", "show_name", "concert_name"]:
        v = event.get(k)
        if v not in (None, ""):
            return str(v).strip()
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, dict):
        for k in ["name", "title"]:
            if raw.get(k):
                return str(raw.get(k)).strip()
    return ""

def _cc_event_venue(event):
    if not isinstance(event, dict):
        return ""
    for k in ["venue", "venue_name", "location_name", "place", "venueName"]:
        v = event.get(k)
        if v not in (None, ""):
            return str(v).strip()
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, dict):
        venue = raw.get("venue")
        if isinstance(venue, dict) and venue.get("name"):
            return str(venue.get("name")).strip()
        embedded = raw.get("_embedded") or {}
        venues = embedded.get("venues") if isinstance(embedded, dict) else None
        if isinstance(venues, list) and venues:
            first = venues[0]
            if isinstance(first, dict) and first.get("name"):
                return str(first.get("name")).strip()
    return ""

def _cc_event_city(event):
    if not isinstance(event, dict):
        return ""
    for k in ["city", "event_city", "venue_city", "location_city"]:
        v = event.get(k)
        if v not in (None, ""):
            return str(v).strip()
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, dict):
        venue = raw.get("venue")
        if isinstance(venue, dict) and venue.get("city"):
            c = venue.get("city")
            if isinstance(c, dict):
                return str(c.get("name") or "").strip()
            return str(c).strip()
        embedded = raw.get("_embedded") or {}
        venues = embedded.get("venues") if isinstance(embedded, dict) else None
        if isinstance(venues, list) and venues:
            first = venues[0]
            if isinstance(first, dict):
                c = first.get("city")
                if isinstance(c, dict):
                    return str(c.get("name") or "").strip()
                if c:
                    return str(c).strip()
    return ""

def _cc_event_date(event):
    return _cc_pick(event, ["event_date", "date", "localDate", "start_date", "date_display", "display_date", "formatted_date", "eventDate", "datetime_local", "event_datetime", "show_date"])

def _cc_event_time(event):
    return _cc_pick(event, ["event_time", "time", "localTime", "start_time", "time_display", "display_time", "formatted_time", "eventTime", "datetime_local", "event_datetime", "show_time"])

def _cc_event_id(event):
    if not isinstance(event, dict):
        return ""
    return str(event.get("event_id") or event.get("external_event_id") or event.get("id") or event.get("source_event_id") or "")








# === V36 training-ready helpers ===
import re
import json as _cc_json
from urllib.parse import quote_plus as _cc_quote_plus
from datetime import datetime as _cc_datetime

def _cc_event_score(event):
    if not isinstance(event, dict):
        return 0.0
    for k in ["model_score", "final_score", "score", "recommendation_score", "copilot_score", "rank_score"]:
        v = event.get(k)
        try:
            if v not in (None, ""):
                return float(v)
        except Exception:
            pass
    return 0.0



def _cc_match_score_label(event):
    try:
        score = float(_cc_event_score(event))
    except Exception:
        score = 0.0
    if score <= 0:
        return "Match Score —"
    if 0 < score <= 1:
        score *= 100.0
    score = max(0.0, min(score, 99.0))
    return f"Match Score {score:.1f}"

def _cc_clean_text(v):
    if v is None:
        return ""
    return str(v).strip()


def _cc_norm_text(v):
    return "".join(ch.lower() for ch in _cc_clean_text(v) if ch.isalnum())


def _cc_pick(event, keys):
    if not isinstance(event, dict):
        return ""
    for key in keys:
        val = event.get(key)
        if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
            return val
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, str):
        try:
            import json as _json
            raw = _json.loads(raw)
        except Exception:
            raw = {}
    if isinstance(raw, dict):
        for key in keys:
            val = raw.get(key)
            if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
                return val
        dates = raw.get("dates") or {}
        if isinstance(dates, dict):
            start = dates.get("start") or {}
            if isinstance(start, dict):
                joined = " ".join(keys).lower()
                if "date" in joined:
                    return start.get("localDate") or start.get("dateTime") or ""
                if "time" in joined:
                    return start.get("localTime") or start.get("dateTime") or ""
        embedded = raw.get("_embedded") or {}
        if isinstance(embedded, dict):
            venues = embedded.get("venues") or []
            if venues and isinstance(venues, list) and isinstance(venues[0], dict):
                joined = " ".join(keys).lower()
                if "venue" in joined:
                    return venues[0].get("name") or ""
                if "city" in joined:
                    city = venues[0].get("city") or {}
                    return city.get("name") if isinstance(city, dict) else city
        classifications = raw.get("classifications") or []
        if isinstance(classifications, list) and classifications:
            c = classifications[0] if isinstance(classifications[0], dict) else {}
            joined = " ".join(keys).lower()
            if "genre" in joined:
                genre = c.get("genre") or {}
                sub = c.get("subGenre") or {}
                seg = c.get("segment") or {}
                return (genre.get("name") if isinstance(genre, dict) else genre) or (sub.get("name") if isinstance(sub, dict) else sub) or (seg.get("name") if isinstance(seg, dict) else seg) or ""
    return ""


def _cc_event_title(event):
    return _cc_pick(event, ["event_name", "title", "name", "artist", "headline", "event"])



def _cc_event_artist(event):
    if not isinstance(event, dict):
        return ""
    for k in ["artist", "artist_name", "primary_artist", "headliner", "performer", "spotify_artist", "spotify_artist_name"]:
        v = event.get(k)
        if v not in (None, ""):
            return str(v).strip()
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, dict):
        performers = raw.get("performers") or raw.get("artists") or []
        if isinstance(performers, list) and performers:
            first = performers[0]
            if isinstance(first, dict) and first.get("name"):
                return str(first.get("name")).strip()
        embedded = raw.get("_embedded") or {}
        attractions = embedded.get("attractions") if isinstance(embedded, dict) else None
        if isinstance(attractions, list) and attractions:
            first = attractions[0]
            if isinstance(first, dict) and first.get("name"):
                return str(first.get("name")).strip()
    title = _cc_event_title(event)
    if not title:
        return ""
    try:
        return re.split(r"\s+(?:w/|with|\+|&|and)\s+", title, flags=re.I)[0].strip()
    except Exception:
        return title.strip()

def _cc_event_venue(event):
    return _cc_pick(event, ["venue", "venue_name", "location", "venueName", "venue_title"])


def _cc_event_city(event):
    return _cc_pick(event, ["city", "venue_city", "metro", "venueCity", "market_city"])


def _cc_event_state(event):
    return _cc_pick(event, ["state", "venue_state", "region", "state_code"])


def _cc_event_date(event):
    return _cc_pick(event, ["event_date", "date", "localDate", "start_date", "date_display", "display_date", "formatted_date", "eventDate", "datetime_local", "event_datetime", "show_date"])


def _cc_event_time(event):
    return _cc_pick(event, ["event_time", "time", "localTime", "start_time", "time_display", "display_time", "formatted_time", "eventTime", "datetime_local", "event_datetime", "show_time"])


def _cc_parse_date_for_sort(event):
    raw = str(_cc_event_date(event) or "")
    if not raw:
        return _cc_datetime.max
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%a, %b %d", "%b %d", "%B %d", "%m/%d/%Y", "%m/%d/%y"]:
        try:
            dt = _cc_datetime.strptime(raw[:10] if fmt.startswith("%Y") else raw.replace("•", " ").strip(), fmt)
            if dt.year == 1900:
                dt = dt.replace(year=_cc_datetime.now().year)
            return dt
        except Exception:
            pass
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", raw)
    if m:
        try:
            return _cc_datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    return _cc_datetime.max


def _cc_event_id(event):
    if not isinstance(event, dict):
        return ""
    return str(event.get("event_id") or event.get("external_event_id") or event.get("id") or event.get("source_event_id") or "")


def _cc_genre(event):
    if not isinstance(event, dict):
        return ""
    for k in ["genre_cluster", "winning_genre_cluster_label", "genre", "genres", "category", "segment"]:
        v = event.get(k)
        if v not in (None, ""):
            if isinstance(v, list):
                return ", ".join(str(x) for x in v if x)
            return str(v).strip()
    return ""

def _cc_match_type(event):
    if not isinstance(event, dict):
        return ""
    for k in ["match_type", "recommendation_type", "fit_label", "badge", "status"]:
        v = event.get(k)
        if v not in (None, ""):
            return str(v).strip()
    if event.get("direct_match") or event.get("is_direct_match") or event.get("spotify_direct_match"):
        return "Direct match"
    return ""

def _cc_score_for_sort(event):
    if not isinstance(event, dict):
        return 0.0
    for k in ["final_score", "score", "model_score", "copilot_score", "rank_score", "recommendation_score", "personalized_score", "spotify_score"]:
        try:
            v = event.get(k)
            if v is not None and str(v).strip() not in ("", "None", "nan"):
                return float(v)
        except Exception:
            pass
    return 0.0


def _cc_price(event):
    if not isinstance(event, dict):
        return None
    for k in ["price_min", "min_price", "lowest_price", "ticket_price", "price", "priceMin", "minPrice"]:
        try:
            v = event.get(k)
            if v is not None and str(v).strip() not in ("", "None", "nan", "0"):
                return float(str(v).replace("$", "").replace(",", ""))
        except Exception:
            pass
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, dict):
        for path in [("stats", "lowest_price"), ("stats", "average_price")]:
            obj = raw
            try:
                for key in path:
                    obj = obj.get(key, {})
                if obj:
                    return float(obj)
            except Exception:
                pass
        prs = raw.get("priceRanges") or []
        if isinstance(prs, list) and prs:
            try:
                return float(prs[0].get("min"))
            except Exception:
                pass
    return None



def _cc_spotify_url(event):
    if not isinstance(event, dict):
        return "https://open.spotify.com/search/concert"
    for k in ["spotify_url", "artist_spotify_url", "spotify_artist_url", "spotify_link", "spotify_href"]:
        v = event.get(k)
        if v not in (None, "", "None"):
            return str(v).strip()
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, dict):
        performers = raw.get("performers") or []
        if isinstance(performers, list):
            for p in performers:
                if isinstance(p, dict):
                    for k in ["url", "spotify_url", "spotify_link"]:
                        u = p.get(k)
                        if u and "spotify" in str(u).lower():
                            return str(u).strip()
    q = _cc_event_artist(event) or _cc_event_title(event) or "concert"
    return "https://open.spotify.com/search/" + urllib.parse.quote(str(q))

def _cc_spotify_label(event):
    url = ""
    if isinstance(event, dict):
        for k in ["spotify_url", "artist_spotify_url", "matched_spotify_url", "spotify_artist_url", "spotify_link", "external_spotify_url"]:
            v = event.get(k)
            if v and str(v).startswith("http"):
                url = str(v)
                break
    if url and "/search/" not in url:
        return "Listen on Spotify"
    return "Search on Spotify"


def _cc_add_spotify_fields(event):
    if not isinstance(event, dict):
        return event
    event = dict(event)
    url = _cc_spotify_url(event)
    event["spotify_url"] = url
    artist = _cc_event_artist(event) or _cc_event_title(event) or "Spotify"
    if "/search/" in url:
        event["spotify_button_label"] = f"Search on Spotify: {artist}"
    else:
        event["spotify_button_label"] = f"Listen on Spotify: {artist}"
    return event

def _dedupe_events_for_display(events):
    """Aggressively merge visually duplicated events across Ticketmaster/SeatGeek."""
    from difflib import SequenceMatcher as _SM
    def clean(v):
        if v is None:
            return ""
        return "".join(ch.lower() for ch in str(v).strip() if ch.isalnum())
    def pick(event, keys):
        if not isinstance(event, dict):
            return ""
        for key in keys:
            val = event.get(key)
            if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
                return val
        raw = event.get("raw_json") or event.get("raw") or {}
        if isinstance(raw, dict):
            for key in keys:
                val = raw.get(key)
                if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
                    return val
            dates = raw.get("dates") or {}
            if isinstance(dates, dict):
                start = dates.get("start") or {}
                if isinstance(start, dict):
                    if "date" in " ".join(keys).lower():
                        return start.get("localDate") or start.get("dateTime") or ""
                    if "time" in " ".join(keys).lower():
                        return start.get("localTime") or start.get("dateTime") or ""
        return ""
    def title(event): return str(pick(event, ["event_name", "title", "name"]))
    def venue(event): return str(pick(event, ["venue", "venue_name", "location"]))
    def city(event): return str(pick(event, ["city", "venue_city"]))
    def date_val(event): return str(pick(event, ["event_date", "date", "localDate", "datetime_local", "event_datetime", "start_date", "date_display"]))[:10]
    def time_val(event): return str(pick(event, ["event_time", "time", "localTime", "datetime_local", "event_datetime", "start_time", "time_display"]))[:5]
    def headliner(event):
        artists = event.get("artists") if isinstance(event, dict) else []
        if artists:
            first = str(artists[0] or "").strip()
            if first:
                return clean(first)
        t = str(title(event)).lower()
        t = re.sub(r".*\bpresents\b", " ", t).strip()
        t = re.split(r"\s+(?:with|w/|feat\.?|featuring|and)\s+|\s[-–—:]\s", t)[0]
        t = re.sub(r"\b(tickets|official|live|concert|tour|event|music|festival|weekend|one|two|the)\b", " ", t)
        return clean(t)
    out, seen_exact = [], set()
    for event in events or []:
        if not isinstance(event, dict):
            continue
        d, tm, v, c, h = date_val(event), time_val(event), clean(venue(event)), clean(city(event)), headliner(event)
        t_clean = clean(title(event))
        exact_key = (d, tm, v, c, h or t_clean[:50])
        if exact_key in seen_exact:
            continue
        duplicate = False
        for kept in out:
            kd, ktm, kv, kc, kh = date_val(kept), time_val(kept), clean(venue(kept)), clean(city(kept)), headliner(kept)
            if d != kd:
                continue
            same_place = (v and kv and v == kv) or (c and kc and c == kc)
            same_time = bool(tm and ktm and tm == ktm)
            if not same_place:
                continue
            title_score = _SM(None, clean(title(event)), clean(title(kept))).ratio()
            head_score = _SM(None, h, kh).ratio() if h and kh else 0.0
            if head_score >= 0.72 or title_score >= 0.72 or (same_time and max(head_score, title_score) >= 0.42):
                duplicate = True
                break
        if duplicate:
            continue
        seen_exact.add(exact_key)
        out.append(event)
    return out


def _cc_dedupe_events_for_display(events):
    return _dedupe_events_for_display(events)


def _cc_event_matches_text(event, query):
    q = str(query or "").strip().lower()
    if not q:
        return True
    hay = " | ".join([
        _cc_event_title(event),
        _cc_event_artist(event),
        _cc_event_venue(event),
        _cc_event_city(event),
        _cc_genre(event),
        _cc_match_type(event),
        str(event.get("why_recommended", "")),
        str(event.get("description", "")),
    ]).lower()
    return all(part in hay for part in q.split())

def _cc_saved_priority_for_event(event, status_by_id=None):
    status_by_id = status_by_id or {}
    eid = _cc_event_id(event)
    raw_status = ""
    if isinstance(event, dict):
        for candidate in [status_by_id.get(eid), event.get("action"), event.get("status"), event.get("preference"), event.get("decision"), event.get("saved_status")]:
            if candidate:
                raw_status = str(candidate).lower()
                break
    if "want" in raw_status or "like" in raw_status or "going" in raw_status:
        return 0
    if "maybe" in raw_status:
        return 1
    if "not" in raw_status or "no" in raw_status or "hide" in raw_status or "dismiss" in raw_status:
        return 3
    return 2



def _cc_apply_discover_filters(events, query="", genre="All genres", match_type=None, sort_mode=None, city_filter="All cities", venue_filter="All venues"):
    """Fast local Discover filtering. No API calls. Keeps memory-first order unless caller sorted already."""
    visible = list(events or [])

    if query:
        visible = [e for e in visible if _cc_event_matches_text(e, query)]

    if city_filter and str(city_filter).lower() not in ["all", "all cities"]:
        cf = str(city_filter).strip().lower()
        visible = [e for e in visible if _cc_event_city(e).strip().lower() == cf]

    if venue_filter and str(venue_filter).lower() not in ["all", "all venues"]:
        vf = str(venue_filter).strip().lower()
        visible = [e for e in visible if _cc_event_venue(e).strip().lower() == vf]

    if genre and str(genre).lower() not in ["all", "all genres"]:
        gf = str(genre).strip().lower()
        visible = [e for e in visible if gf in _cc_genre(e).strip().lower()]

    # Keep saved-memory-first order from the ranker. Only apply score/date sorting if old code explicitly asks.
    sm = str(sort_mode or "memory first").lower()
    if "score" in sm:
        visible = sorted(visible, key=lambda e: _cc_event_score(e), reverse=True)
    elif "soon" in sm or "date" in sm:
        def _d(e):
            for k in ["event_datetime", "datetime", "date_time", "start_time", "event_date", "date"]:
                v = e.get(k) if isinstance(e, dict) else None
                if v:
                    return str(v)
            return "9999"
        visible = sorted(visible, key=_d)
    return visible

def _cc_limit_visible_events(events, key="discover", page_size=40):
    events = list(events or [])
    total = len(events)
    if total <= page_size:
        return events
    sig = f"{key}:{total}:" + "|".join([_cc_event_title(e) for e in events[:5]])
    sig_key = f"{key}_visible_signature"
    limit_key = f"{key}_visible_limit"
    if st.session_state.get(sig_key) != sig:
        st.session_state[sig_key] = sig
        st.session_state[limit_key] = page_size
    limit = int(st.session_state.get(limit_key, page_size))
    limit = max(page_size, min(limit, total))
    st.caption(f"Showing {limit} of {total} results. Search and filters still apply to the full loaded list.")
    if limit < total:
        if st.button(f"Load {min(page_size, total - limit)} more", key=f"{key}_load_more_btn"):
            st.session_state[limit_key] = min(total, limit + page_size)
            st.rerun()
    return events[:limit]

def _cc_sort_saved_first(events, status_by_id=None):
    return sorted(list(events or []), key=lambda e: (_cc_saved_priority_for_event(e, status_by_id), -_cc_score_for_sort(e), _cc_parse_date_for_sort(e)))

def _cc_api_keyword(keyword):
    """Keep external API search broad when the sidebar contains a multi-intent taste prompt.
    The typed keyword still works inside Discover search; this prevents Ticketmaster/SeatGeek from returning zero for strings like 'country, electronic, Ella Langley'.
    """
    kw = str(keyword or "").strip()
    if not kw:
        return None
    if "," in kw or ";" in kw or " and " in kw.lower():
        return None
    if len(kw.split()) > 4:
        return None
    return kw


# === end V36 helpers ===


def _dedupe_events_for_display(events):
    """Aggressively merge visually duplicated events across Ticketmaster/SeatGeek."""
    from difflib import SequenceMatcher as _SM
    def clean(v):
        if v is None:
            return ""
        return "".join(ch.lower() for ch in str(v).strip() if ch.isalnum())
    def pick(event, keys):
        if not isinstance(event, dict):
            return ""
        for key in keys:
            val = event.get(key)
            if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
                return val
        raw = event.get("raw_json") or event.get("raw") or {}
        if isinstance(raw, dict):
            for key in keys:
                val = raw.get(key)
                if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
                    return val
            dates = raw.get("dates") or {}
            if isinstance(dates, dict):
                start = dates.get("start") or {}
                if isinstance(start, dict):
                    if "date" in " ".join(keys).lower():
                        return start.get("localDate") or start.get("dateTime") or ""
                    if "time" in " ".join(keys).lower():
                        return start.get("localTime") or start.get("dateTime") or ""
        return ""
    def title(event): return str(pick(event, ["event_name", "title", "name"]))
    def venue(event): return str(pick(event, ["venue", "venue_name", "location"]))
    def city(event): return str(pick(event, ["city", "venue_city"]))
    def date_val(event): return str(pick(event, ["event_date", "date", "localDate", "datetime_local", "event_datetime", "start_date", "date_display"]))[:10]
    def time_val(event): return str(pick(event, ["event_time", "time", "localTime", "datetime_local", "event_datetime", "start_time", "time_display"]))[:5]
    def headliner(event):
        artists = event.get("artists") if isinstance(event, dict) else []
        if artists:
            first = str(artists[0] or "").strip()
            if first:
                return clean(first)
        t = str(title(event)).lower()
        t = re.sub(r".*\bpresents\b", " ", t).strip()
        t = re.split(r"\s+(?:with|w/|feat\.?|featuring|and)\s+|\s[-–—:]\s", t)[0]
        t = re.sub(r"\b(tickets|official|live|concert|tour|event|music|festival|weekend|one|two|the)\b", " ", t)
        return clean(t)
    out, seen_exact = [], set()
    for event in events or []:
        if not isinstance(event, dict):
            continue
        d, tm, v, c, h = date_val(event), time_val(event), clean(venue(event)), clean(city(event)), headliner(event)
        t_clean = clean(title(event))
        exact_key = (d, tm, v, c, h or t_clean[:50])
        if exact_key in seen_exact:
            continue
        duplicate = False
        for kept in out:
            kd, ktm, kv, kc, kh = date_val(kept), time_val(kept), clean(venue(kept)), clean(city(kept)), headliner(kept)
            if d != kd:
                continue
            same_place = (v and kv and v == kv) or (c and kc and c == kc)
            same_time = bool(tm and ktm and tm == ktm)
            if not same_place:
                continue
            title_score = _SM(None, clean(title(event)), clean(title(kept))).ratio()
            head_score = _SM(None, h, kh).ratio() if h and kh else 0.0
            if head_score >= 0.72 or title_score >= 0.72 or (same_time and max(head_score, title_score) >= 0.42):
                duplicate = True
                break
        if duplicate:
            continue
        seen_exact.add(exact_key)
        out.append(event)
    return out


# === end HOTFIX V34 helpers ===

st.set_page_config(page_title="Encore AI V40 — Stable Final", page_icon="🎧", layout="wide")
init_db()

for key, default in {
    "browser_session_id": str(uuid.uuid4()),
    "recommendation_run": None,
    "hidden_event_ids": set(),
    "taste_cache": {},
    "copilot_question": "",
    "copilot_response": None,
    "night_plan_result": None,
    "elite_copilot_result": None,
    "plan_event_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_data(ttl=900, show_spinner=False)
def cached_event_search(
    city,
    state,
    country,
    radius,
    size,
    keyword,
    use_ticketmaster,
    use_seatgeek,
    use_songkick,
    ticketmaster_pages,
    seatgeek_pages,
    start_date,
    end_date,
    price_enrichment_limit,
    max_targeted_artists,
    venue_name,
    targeted_artists,
):
    return search_all_sources(
        city=city,
        state_code=state,
        country_code=country,
        radius=radius,
        size=size,
        keyword=_cc_api_keyword(keyword),
        use_ticketmaster=use_ticketmaster,
        use_seatgeek=use_seatgeek,
        use_songkick=use_songkick,
        ticketmaster_pages=ticketmaster_pages,
        seatgeek_pages=seatgeek_pages,
        start_date=start_date,
        end_date=end_date,
        venue_name=venue_name or None,
        targeted_artist_names=list(targeted_artists),
        max_targeted_artists=max_targeted_artists,
        smart_artist_search=True,
        price_enrichment_limit=price_enrichment_limit,
    )


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --ink:#171b26; --muted:#687083; --soft:#f4f5f9; --line:#e5e8f0;
  --coral:#ff5954; --coral-soft:#fff0ee; --amber:#df8a19; --green:#1fb857;
}
html, body, [class*="css"] {font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}
.block-container {padding-top: 1.05rem; max-width: 1500px;}
body {background:#f2f3f7;}
/* Clean demo header */
.new-hero {padding:26px 32px;border-radius:24px;background:#fff;color:var(--ink);box-shadow:0 14px 36px rgba(17,24,39,.07);margin:0 0 1.1rem;border:1px solid #e7eaf1;position:relative;overflow:hidden;}
.new-hero::before {content:'';position:absolute;top:-95px;right:-40px;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle, rgba(255,89,84,.13), transparent 66%);pointer-events:none;}
.hero-eyebrow {font-size:11.5px;font-weight:850;letter-spacing:.18em;text-transform:uppercase;color:var(--coral);margin-bottom:12px;position:relative;z-index:1;}
.hero-h1 {font-family:'Bricolage Grotesque',Inter,sans-serif;font-weight:850;font-size:38px;line-height:1.04;letter-spacing:-.045em;margin:0 0 10px;max-width:820px;position:relative;z-index:1;color:var(--ink);}
.hero-sub {font-size:15px;line-height:1.5;color:#687083;margin:0 0 18px;max-width:760px;position:relative;z-index:1;}
.filter-summary {display:flex;align-items:center;gap:10px;flex-wrap:wrap;position:relative;z-index:1;margin-top:16px;}
.filter-pill {display:inline-flex;align-items:center;gap:7px;border-radius:999px;background:#f6f7fb;border:1px solid #e7eaf1;color:#303747;font-size:13px;font-weight:800;padding:8px 12px;}
.filter-pill .mini-dot {width:7px;height:7px;border-radius:99px;background:var(--coral);display:inline-block;}
.demo-note {font-size:13.5px;color:#7a8295;margin-top:12px;position:relative;z-index:1;}
.stats-strip {display:flex;align-items:center;gap:12px 26px;padding:10px 4px 18px;flex-wrap:wrap;}
.stat-num {font-family:'Bricolage Grotesque',Inter,sans-serif;font-weight:850;font-size:25px;letter-spacing:-.02em;color:var(--ink);}
.stat-num-coral {color:var(--coral);}
.stat-lbl {font-size:13.5px;color:#5f6677;}
.stat-divider {width:1px;height:22px;background:#d9dce5;}
.ranking-pill {margin-left:auto;display:inline-flex;align-items:center;gap:8px;padding:7px 14px;border-radius:999px;background:#fff;border:1px solid #e6e8ef;font-size:12.5px;font-weight:750;color:#5a6171;box-shadow:0 8px 20px rgba(17,24,39,.04);}
.section-head {font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.55rem; font-weight:850; color:var(--ink); margin:.35rem 0 .12rem 0;letter-spacing:-.035em;}
.section-sub {font-size:.94rem; color:#7a8295; margin:0 0 .9rem 0;}

/* Premium horizontal event cards */

/* Native Streamlit bordered containers become the premium cards. */
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:28px!important;border:1px solid #e6e8f0!important;box-shadow:0 18px 40px rgba(17,24,39,.07)!important;background:#fff!important;}
div[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:28px!important;}
.premium-shell {background:#fff;border:1px solid #e6e8f0;border-radius:28px;padding:1.15rem 1.25rem;box-shadow:0 18px 40px rgba(17,24,39,.07);margin:0 0 1.05rem 0;}
.poster-wrap {width:190px;height:190px;border-radius:22px;overflow:hidden;background:linear-gradient(135deg,#eee7d8,#f7f2e8);display:flex;align-items:center;justify-content:center;border:1px solid #eee6d8;}
.poster-wrap img {width:190px;height:190px;object-fit:cover;display:block;}
.poster-fallback {width:190px;height:190px;border-radius:22px;background:repeating-linear-gradient(135deg,#f3ecdf,#f3ecdf 12px,#eee6d8 12px,#eee6d8 18px);display:flex;align-items:center;justify-content:center;color:#ab9b7d;font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;position:relative;}
.poster-fallback:before {content:"";width:42px;height:42px;border-radius:50%;border:3px solid #c8b78e;position:absolute;top:54px;}
.card-topline {display:flex;align-items:center;gap:.8rem;margin-bottom:.4rem;}
.rank-chip {display:inline-flex;align-items:center;justify-content:center;background:var(--coral-soft);color:var(--coral);font-weight:850;border-radius:10px;padding:.28rem .48rem;font-size:.88rem;}
.card-date {color:#5f6677;font-size:1.02rem;font-weight:700;}
.card-title {font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.75rem;line-height:1.02;font-weight:800;color:var(--ink);letter-spacing:-.04em;margin:.05rem 0 .26rem;}
.card-venue {color:#626a7b;font-size:1.05rem;margin-bottom:.85rem;}
/* Signal row: fixed, clean alignment across every card. */
.signal-row {display:flex;align-items:center;gap:.48rem;row-gap:.45rem;margin:.15rem 0 .78rem;width:100%;flex-wrap:wrap;}
.signal-pill {display:inline-flex;align-items:center;justify-content:center;gap:.42rem;border-radius:999px;background:#f5f6fa;border:1px solid #edf0f5;color:#343946;font-size:.88rem;font-weight:800;padding:.38rem .72rem;white-space:nowrap;line-height:1.05;min-height:32px;}
.signal-dot {width:8px;height:8px;min-width:8px;border-radius:999px;display:inline-block;background:#b8bfcd;}
.dot-coral {background:var(--coral);} .dot-green {background:var(--green);} .dot-amber {background:var(--amber);} .dot-blue {background:#4d7cff;}
.spotify-pill {display:inline-flex;align-items:center;justify-content:center;gap:.42rem;border-radius:999px;background:#fff;border:1px solid #dfe3ec;color:#5a6171;font-size:.84rem;font-weight:850;padding:.38rem .72rem;text-decoration:none;margin-left:auto;white-space:nowrap;line-height:1.05;min-height:32px;max-width:100%;overflow:hidden;text-overflow:ellipsis;}
.why-note {border-left:3px solid #ffd0ca;padding:.1rem 0 .1rem 1.05rem;margin:.55rem 0 .85rem;}
.why-label {font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:#a1a8b7;font-weight:900;margin-bottom:.2rem;}
.why-copy {font-size:1.0rem;color:#5f6677;line-height:1.55;}
.action-divider {width:1px;height:34px;background:#e9ecf2;margin:0 .25rem;}
.action-hint {font-size:.76rem;color:#858c9c;margin-top:.22rem;}
.clean-tag {display:inline-flex;align-items:center;padding:.26rem .62rem;margin:.08rem .14rem .08rem 0;border-radius:999px;font-size:.76rem;background:#f6f7fb;border:1px solid #e8ebf2;color:#4a5162;font-weight:750;}
.badge {display:inline-block;padding:.22rem .58rem;margin:.1rem .18rem .1rem 0;border-radius:999px;font-size:.75rem;background:#f1f3f6;border:1px solid #e1e4ea;color:#3a4050;}
.badge-direct {background:#eef0ff;border-color:#cdd2ff;color:#343a8d;}
.badge-price {background:#edf9f1;border-color:#bce4c7;color:#225d34;font-weight:700;}
.badge-warn {background:#fff7eb;border-color:#f2d2a3;color:#784c0e;}
.badge-group {background:#fff0f7;border-color:#f7c3dc;color:#8a2f5d;}
.status-pill {display:inline-block;background:#effaf4;border:1px solid #bfe7cd;color:#24633a;border-radius:999px;padding:.3rem .7rem;font-size:.82rem;font-weight:700;}
.status-base {background:#f4f5f8;border-color:#dfe2e8;color:#555d6c;}
.timeline-card {background:#f8f9fb;border:1px solid #e4e7ed;border-radius:16px;padding:.8rem;min-height:118px;}
.timeline-time {font-size:.77rem;color:#7a8190;text-transform:uppercase;font-weight:700;}
.timeline-step {font-size:1rem;font-weight:800;color:#252b39;margin:.12rem 0;}
.timeline-detail {font-size:.84rem;color:#626a7b;}
.place-card {background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:.8rem;height:100%;}
[data-testid="stSidebar"] {background:#f3f4f8;border-right:1px solid #e1e4ea;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#34394a;}
/* Make primary link buttons feel like the coral CTA. */
div[data-testid="stLinkButton"] a[kind="primary"] {background:var(--coral)!important;border-color:var(--coral)!important;border-radius:14px!important;font-weight:850!important;box-shadow:0 8px 16px rgba(255,89,84,.22)!important;}
div[data-testid="stButton"] button[kind="secondary"] {border-radius:14px!important;border:1px solid #dfe3ec!important;color:#2b303c!important;background:#fff!important;font-weight:800!important;}

/* Card alignment fixes */
div[data-testid="stHorizontalBlock"] {align-items:stretch;}
.poster-wrap,.poster-fallback,.poster-wrap img {min-width:190px;max-width:190px;min-height:190px;max-height:190px;}
.poster-wrap img {object-position:center center;}
.signal-pill.no-dot {padding-left:.72rem;}
.spotify-pill:hover {border-color:#bce7ca;background:#f7fff9;color:#1a7f42;text-decoration:none;}
.spotify-helper {font-size:.76rem;color:#8a92a3;margin-top:.18rem;}
div[class*="st-key-no_"] button {color:#c9403c!important;border-color:#ffd2ce!important;background:#fff8f7!important;}
div[class*="st-key-no_"] button:hover {border-color:#f15a52!important;background:#fff0ee!important;color:#ab302c!important;}
.discover-head {display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin:.25rem 0 .8rem;}
.discover-title {font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.65rem;font-weight:850;letter-spacing:-.035em;color:var(--ink);margin:0;}
.discover-copy {color:#687083;font-size:.95rem;margin:.12rem 0 0;}

.model-status-strip {display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:6px 0 18px;}
.model-status-detail {font-size:13px;color:#7a8295;}
.shortlist-summary {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:.6rem 0 1rem;}
.shortlist-summary-card {background:#fff;border:1px solid #e6e8ef;border-radius:18px;padding:14px 16px;box-shadow:0 10px 24px rgba(17,24,39,.04);}
.shortlist-summary-card .num {font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:28px;font-weight:850;color:var(--ink);line-height:1;}
.shortlist-summary-card .lbl {font-size:12px;color:#7a8295;font-weight:800;text-transform:uppercase;letter-spacing:.08em;margin-top:6px;}
.shortlist-card {background:#fff;border:1px solid #e6e8ef;border-radius:22px;padding:18px 20px;margin:0 0 14px;box-shadow:0 12px 30px rgba(17,24,39,.045);}
.shortlist-title {font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.35rem;font-weight:850;letter-spacing:-.025em;color:var(--ink);margin-bottom:.35rem;}
.shortlist-meta {color:#606879;font-size:.96rem;margin:.18rem 0 .55rem;}
.shortlist-reason {color:#687083;font-size:.92rem;line-height:1.45;margin:.35rem 0 .8rem;}
.agent-context {background:#f8f9fc;border:1px solid #e7eaf1;border-radius:18px;padding:14px 16px;margin:.6rem 0 1rem;color:#384052;}
.agent-context strong {color:#171b26;}
.copilot-chip-row {display:flex;gap:8px;flex-wrap:wrap;margin:.25rem 0 1rem;}
.copilot-note {font-size:13px;color:#7a8295;margin:.35rem 0 .9rem;}

@media (max-width: 900px) {.new-hero{padding:30px 24px}.hero-h1{font-size:32px}.search-bar{display:block}.search-div{display:none}.search-cta{width:100%;margin-top:6px}.ranking-pill{margin-left:0}.discover-head{display:block}}


.model-status-strip {display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:6px 0 18px;}
.model-status-detail {font-size:13px;color:#7a8295;}
.shortlist-summary {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:.6rem 0 1rem;}
.shortlist-summary-card {background:#fff;border:1px solid #e6e8ef;border-radius:18px;padding:14px 16px;box-shadow:0 10px 24px rgba(17,24,39,.04);}
.shortlist-summary-card .num {font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:28px;font-weight:850;color:var(--ink);line-height:1;}
.shortlist-summary-card .lbl {font-size:12px;color:#7a8295;font-weight:800;text-transform:uppercase;letter-spacing:.08em;margin-top:6px;}
.shortlist-card {background:#fff;border:1px solid #e6e8ef;border-radius:22px;padding:18px 20px;margin:0 0 14px;box-shadow:0 12px 30px rgba(17,24,39,.045);}
.shortlist-title {font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.35rem;font-weight:850;letter-spacing:-.025em;color:var(--ink);margin-bottom:.35rem;}
.shortlist-meta {color:#606879;font-size:.96rem;margin:.18rem 0 .55rem;}
.shortlist-reason {color:#687083;font-size:.92rem;line-height:1.45;margin:.35rem 0 .8rem;}
.agent-context {background:#f8f9fc;border:1px solid #e7eaf1;border-radius:18px;padding:14px 16px;margin:.6rem 0 1rem;color:#384052;}
.agent-context strong {color:#171b26;}
.copilot-chip-row {display:flex;gap:8px;flex-wrap:wrap;margin:.25rem 0 1rem;}
.copilot-note {font-size:13px;color:#7a8295;margin:.35rem 0 .9rem;}

@media (max-width: 900px) {.poster-wrap,.poster-fallback,.poster-wrap img{width:130px;height:130px;min-width:130px;max-width:130px;min-height:130px;max-height:130px}.card-title{font-size:1.35rem}.card-date{font-size:.92rem}.premium-shell{padding:.95rem}.signal-row{display:flex;flex-wrap:wrap}.spotify-pill{justify-self:start}}
.mini-score{font-size:.76rem;color:#64748b;margin-top:.55rem;font-weight:700}.mini-card{border:1px solid rgba(15,23,42,.10);border-radius:14px;padding:14px;background:#fff;min-height:104px;box-shadow:0 8px 24px rgba(15,23,42,.04)}.mini-card span{color:#64748b;font-size:.85rem}</style>
""",
    unsafe_allow_html=True,
)


# ---------------- Formatting helpers ----------------
def format_event_date(value):
    if not value:
        return "Date TBD"
    try:
        parsed = datetime.strptime(str(value)[:10], "%Y-%m-%d")
        return parsed.strftime("%a, %b %d").replace(" 0", " ")
    except Exception:
        return str(value)


def format_event_time(value):
    if not value:
        return "Time TBD"
    text = str(value).strip()
    for pattern in ("%H:%M:%S", "%H:%M", "%I:%M %p"):
        try:
            parsed = datetime.strptime(text[:8] if pattern == "%H:%M:%S" else text, pattern)
            return parsed.strftime("%I:%M %p").lstrip("0")
        except Exception:
            continue
    return text


def format_when(event):
    return f"{format_event_date(event.get('date'))} · {format_event_time(event.get('time'))}"


def render_hero(city: str, state: str, radius: int, start_date: str, end_date: str, keyword: str = "") -> None:
    st.markdown("""
    <div class="new-hero">
      <div class="hero-eyebrow">Personalized to your Spotify</div>
      <div class="hero-h1">Encore AI is ranking shows for <span style="color:#ff5954;">your taste</span>.</div>
      <div class="hero-sub">Use the working filters in the left sidebar, then click <b>Get recommendations</b>. The promoted feedback model stays active automatically.</div>
    </div>
    """, unsafe_allow_html=True)

def render_stats_strip(ranked_count: int, direct_count: int, price_coverage: int, model_active: bool, bundle: Dict[str, Any] | None = None) -> None:
    """V29: cleaner Simple Copilot with request-window aware picks and fewer tabs."""
    if model_active:
        weight = float((bundle or {}).get("recommended_model_weight") or 0.20)
        ranking_note = f"Feedback model active · {weight:.0%} blend"
        detail = "Personalized reranker is safely blended into Spotify taste ranking."
    else:
        ranking_note = "Spotify taste ranking active"
        detail = "A promoted feedback model will activate after it passes evaluation."
    st.markdown(f"""
    <div class="model-status-strip">
      <span class="ranking-pill"><span style="width:7px;height:7px;border-radius:50%;background:#1f9d57;display:inline-block;"></span>{escape(ranking_note)}</span>
      <span class="model-status-detail">{escape(detail)}</span>
    </div>
    """, unsafe_allow_html=True)




def load_playlist_preferences(user_id: str) -> pd.DataFrame:
    """Latest saved/hidden preferences for the user: Want, Maybe, and Not a Fit."""
    try:
        return load_user_latest_preferences(user_id)
    except Exception:
        return pd.DataFrame()


def preference_maps(user_id: str):
    prefs = load_playlist_preferences(user_id)
    status_by_id = {}
    if not prefs.empty and "event_id" in prefs.columns and "action" in prefs.columns:
        for row in prefs.to_dict(orient="records"):
            if row.get("event_id"):
                status_by_id[str(row.get("event_id"))] = row.get("action")
    hidden_ids = {eid for eid, action in status_by_id.items() if action == "not_for_me"}
    return prefs, status_by_id, hidden_ids


def preference_label(action: str | None) -> str | None:
    if action == "want_to_go":
        return "Saved · Want"
    if action == "maybe":
        return "Saved · Maybe"
    if action == "not_for_me":
        return "Hidden · Not a Fit"
    return None


def unique_events_by_id(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for event in events:
        key = str(event.get("event_id") or event.get("event_name") or len(output))
        if key in seen:
            continue
        seen.add(key)
        output.append(event)
    return output

def build_calendar_ics(event: Dict[str, Any]) -> str:
    title = str(event.get("event_name") or "Concert")
    venue = str(event.get("venue") or "Venue TBD")
    city_state = f"{event.get('city') or ''}, {event.get('state') or ''}".strip(" ,")
    location = f"{venue}, {city_state}".strip(" ,")
    event_date = str(event.get("date") or date.today().isoformat())[:10]
    event_time = str(event.get("time") or "19:00:00")[:8]
    if len(event_time) == 5:
        event_time += ":00"
    try:
        start_dt = datetime.strptime(f"{event_date} {event_time}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        start_dt = datetime.strptime(event_date, "%Y-%m-%d").replace(hour=19, minute=0, second=0)
    end_dt = start_dt + timedelta(hours=3)
    uid = str(event.get("event_id") or uuid.uuid4()).replace("@", "-")
    description = str(event.get("why_recommended") or event.get("why_artist_match") or "Recommended by Encore AI")

    def esc_ics(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\n", "\\n")
        )

    return "\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Encore AI//V21//EN",
        "BEGIN:VEVENT",
        f"UID:{uid}@encore-ai.local",
        f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}",
        f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}",
        f"SUMMARY:{esc_ics(title)}",
        f"LOCATION:{esc_ics(location)}",
        f"DESCRIPTION:{esc_ics(description)}",
        "END:VEVENT",
        "END:VCALENDAR",
    ])


def parse_group_artists(text: str) -> List[str]:
    return [value.strip() for value in (text or "").replace("\n", ",").split(",") if value.strip()]


def event_links(event):
    return [url for url in (event.get("all_urls") or [event.get("url")]) if url]



def price_label(event):
    minimum = event.get("min_price")
    maximum = event.get("max_price")
    median = event.get("median_price")
    average = event.get("average_price")
    source = str(event.get("price_source") or "").strip()
    suffix = ""
    if source:
        if "SeatGeek" in source:
            suffix = " · SeatGeek"
        elif "Ticketmaster" in source:
            suffix = " · Ticketmaster"
    if isinstance(minimum, (int, float)):
        if isinstance(maximum, (int, float)) and maximum > minimum and maximum <= minimum * 4:
            return f"${minimum:.0f}–${maximum:.0f}{suffix}", "badge-price"
        return f"From ${minimum:.0f}{suffix}", "badge-price"
    if isinstance(median, (int, float)):
        return f"Typical ${median:.0f}{suffix}", "badge-price"
    if isinstance(average, (int, float)):
        return f"Avg ${average:.0f}{suffix}", "badge-price"
    return "Price not listed", "badge-muted"

def model_status():
    bundle = load_feedback_model("current")
    if bundle:
        return {
            "active": True,
            "label": "Feedback model active",
            "detail": f"Safely blended with Spotify ranking · trained on {bundle.get('n_rows', 0)} ratings",
            "bundle": bundle,
        }
    return {
        "active": False,
        "label": "Spotify taste model",
        "detail": "A trained feedback model will activate automatically after it passes evaluation.",
        "bundle": None,
    }


def annotate_group_fit(events, primary_artists, second_artists, primary_name, second_name):
    primary_set = {(artist.get("artist") or "").lower() for artist in primary_artists if artist.get("artist")}
    second_set = {artist.lower() for artist in second_artists}
    output = []
    for event in events:
        item = dict(event)
        artists = {(artist or "").lower() for artist in (item.get("artists") or [])}
        primary_match = sorted(primary_set & artists)
        second_match = sorted(second_set & artists)
        if primary_match and second_match:
            label, boost = "Both listeners", 22
        elif second_match:
            label, boost = f"{second_name} match", 13
        elif primary_match:
            label, boost = f"{primary_name} match", 10
        else:
            label, boost = "Shared discovery", min(float(item.get("genre_score") or 0) / 4, 8)
        item["group_fit_label"] = label
        item["group_fit_score"] = round(float(item.get("final_score") or 0) + boost, 2)
        output.append(item)
    return sorted(output, key=lambda value: value.get("group_fit_score", 0), reverse=True)


def render_badges(event, extra_badge=None):
    price, price_class = price_label(event)
    lane = _cc_display_lane(event)
    confidence = event.get("match_confidence") or "Taste match"
    source_count = human_source_count(event)
    multi = " · multi-source confirmed" if source_count > 1 else ""
    group_html = f'<span class="badge badge-group">{extra_badge}</span>' if extra_badge else ""
    plural = "s" if source_count != 1 else ""
    st.markdown(
        f'<span class="badge badge-direct">{confidence}</span>'
        f'{group_html}'
        f'<span class="badge {price_class}">{price}</span>'
        f'<span class="badge">{lane}</span>'
        '',
        unsafe_allow_html=True,
    )


def _feedback_reason_value(key: str) -> list[str]:
    value = st.session_state.get(key, [])
    return value if isinstance(value, list) else ([] if not value else [str(value)])


CITY_STATE_DEFAULTS = {
    "austin": "TX", "dallas": "TX", "houston": "TX", "san antonio": "TX", "fort worth": "TX",
    "nashville": "TN", "denver": "CO", "chicago": "IL", "new york": "NY", "brooklyn": "NY",
    "los angeles": "CA", "san francisco": "CA", "san diego": "CA", "seattle": "WA", "miami": "FL",
    "atlanta": "GA", "charleston": "SC", "boston": "MA", "philadelphia": "PA", "washington": "DC",
    "portland": "OR", "phoenix": "AZ", "las vegas": "NV", "minneapolis": "MN", "milwaukee": "WI",
    "new orleans": "LA", "raleigh": "NC", "charlotte": "NC", "salt lake city": "UT", "kansas city": "MO",
}
CITY_OPTIONS = [city.title() if city != "new york" else "New York" for city in CITY_STATE_DEFAULTS.keys()]
CITY_OPTIONS = ["Austin"] + [city for city in CITY_OPTIONS if city != "Austin"]
STATE_OPTIONS = ["TX", "TN", "CO", "IL", "NY", "CA", "WA", "FL", "GA", "SC", "MA", "PA", "DC", "OR", "AZ", "NV", "MN", "WI", "LA", "NC", "UT", "MO"]

AUSTIN_VENUES = [
    "All venues", "Moody Center ATX", "ACL Live", "Stubb's Waller Creek Amphitheater",
    "Mohawk", "Emo's Austin", "Scoot Inn", "3TEN ACL Live", "Antone's Nightclub",
    "The Parish", "Empire Control Room & Garage", "Come and Take It Live",
    "Germania Insurance Amphitheater", "Bass Concert Hall", "Custom venue...",
]

FEEDBACK_REASON_OPTIONS = [
    "Artist match was right", "Genre / vibe was right", "Great discovery", "Good venue",
    "Good price", "Wrong artist match", "Wrong genre / vibe", "Too expensive",
    "Bad date / time", "Bad venue", "Too far", "Already seen them",
]


def infer_state_for_city(city_value: str, fallback: str = "TX") -> str:
    return CITY_STATE_DEFAULTS.get(str(city_value or "").strip().lower(), fallback or "TX")


def display_recommendation_mode(style: str) -> str:
    return style or "Best overall"


def human_source_count(event):
    count = int(event.get("source_count") or len(event.get("sources") or []) or 1)
    return count


def spotify_link_html(event):
    links = event.get("artist_spotify_urls") or []
    if not links:
        return ""
    parts = []
    for item in links[:2]:
        artist = item.get("artist") or "Artist"
        url = item.get("url")
        if url:
            parts.append(f'<a class="spotify-link" href="{url}" target="_blank">▶ {artist} on Spotify</a>')
    return "".join(parts)



def _cc_display_lane(event):
    lane = event.get("winning_genre_cluster_label") or event.get("genre") or event.get("subgenre")
    lane = str(lane or "").strip()
    bad = {"", "none", "nan", "music", "unclear taste lane", "undefined", "miscellaneous"}
    if lane.lower() in bad:
        lane = str(event.get("subgenre") or event.get("genre") or "").strip()
    if not lane or lane.lower() in bad:
        lane = "Music discovery"
    return lane


def reason_tags_html(event):
    tags = event.get("reason_tags") or []
    if not tags:
        tags = [event.get("match_confidence") or "Taste match", event.get("winning_genre_cluster_label") or event.get("genre") or "Music"]
    return "".join([f'<span class="clean-tag">{tag}</span>' for tag in tags[:5] if tag])

def render_event_card(event: Dict[str, Any], idx: int, section: str, user, session_id, extra_badge=None):
    event = _cc_add_spotify_fields(dict(event or {}))
    title = event.get('event_name') or 'Untitled event'
    venue_line = f"{event.get('venue') or 'Venue TBD'} · {event.get('city') or ''}, {event.get('state') or ''}".strip(" ·,")
    when = format_when(event)
    price_text, price_class = price_label(event)
    lane = _cc_display_lane(event)
    confidence = event.get("match_confidence") or "Taste match"
    why = event.get("why_artist_match") or event.get("why_recommended") or "This event matches your broader listening profile."
    lane_copy = event.get("why_taste_lane") or ""
    reason_key = f"reason_{section}_{session_id}_{event.get('event_id')}_{idx}"
    links = event_links(event)

    with st.container(border=True):
        poster_col, content_col = st.columns([0.18, 0.82], vertical_alignment="top")
        with poster_col:
            if event.get("image_url"):
                st.markdown(
                    f'<div class="poster-wrap"><img src="{escape(str(event.get("image_url")))}" alt="event poster"></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="poster-fallback">Event Poster</div>', unsafe_allow_html=True)

        with content_col:
            st.markdown(
                f"""
                <div class="card-topline">
                  <span class="rank-chip">#{idx}</span>
                  <span class="card-date">{escape(when)}</span>
                </div>
                <div class="card-title">{escape(str(title))}</div>
                <div class="card-venue">{escape(venue_line)}</div>
                """,
                unsafe_allow_html=True,
            )

            spotify_links = event.get("artist_spotify_urls") or []
            spotify_pill = ""
            if spotify_links:
                item = spotify_links[0]
                if item.get("url"):
                    spotify_pill = (
                        f'<a class="spotify-pill" href="{escape(str(item.get("url")))}" target="_blank">'
                        f'<span class="signal-dot dot-green"></span>Listen on Spotify: {escape(str(item.get("artist") or "Artist"))}</a>'
                    )
            if not spotify_pill:
                _sp_url = event.get('spotify_url') or _cc_spotify_url(event)
                _sp_label = event.get('spotify_label') or _cc_spotify_label(event)
                spotify_pill = f'<a class="spotify-pill" href="{escape(str(_sp_url))}" target="_blank"><span class="signal-dot dot-green"></span>{escape(str(_sp_label))}</a>'
            price_dot = "dot-green" if price_class == "badge-price" else "dot-amber"
            group_html = f'<span class="signal-pill"><span class="signal-dot dot-blue"></span>{escape(str(extra_badge))}</span>' if extra_badge else ""
            match_score_html = f'<span class="signal-pill badge-price">{escape(_cc_match_score_label(event))}</span>'
            st.markdown(
                f"""
                <div class="signal-row">
                  {match_score_html}
                  <span class="signal-pill"><span class="signal-dot dot-coral"></span>{escape(str(confidence))}</span>
                  {group_html}
                  <span class="signal-pill no-dot">{escape(str(lane))}</span>
                  <span class="signal-pill"><span class="signal-dot {price_dot}"></span>{escape(str(price_text))}</span>
                  {spotify_pill}
                </div>
                <div class="why-note">
                  <div class="why-label">Why it fits</div>
                  <div class="why-copy">{escape(str((why + ' ' + lane_copy).strip()))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.popover("Feedback reasons"):
                st.multiselect(
                    "Pick any that apply",
                    FEEDBACK_REASON_OPTIONS,
                    key=reason_key,
                    help="Optional, but helpful. It tells the model whether the music match, price, venue, or timing caused your choice.",
                )
                st.caption("Select multiple reasons if useful. The main label is still Want / Maybe / Not a Fit.")

            action_cols = st.columns([1.0, 0.07, 1.0, 0.75, 1.0, 2.2], vertical_alignment="center")
            with action_cols[0]:
                if links:
                    ticket_label = "Tickets / live price →" if price_text == "Price not listed" else "Tickets →"
                    st.link_button(ticket_label, links[0], type="primary", use_container_width=True)
                else:
                    st.button("Tickets →", key=f"tickets_disabled_{section}_{session_id}_{event.get('event_id')}_{idx}", disabled=True, use_container_width=True)
                try:
                    st.link_button(event.get('spotify_label') or _cc_spotify_label(event), event.get('spotify_url') or _cc_spotify_url(event), use_container_width=True)
                except Exception:
                    pass
            with action_cols[1]:
                st.markdown('<div class="action-divider"></div>', unsafe_allow_html=True)
            with action_cols[2]:
                if st.button("Want to go", key=f"want_{section}_{session_id}_{event.get('event_id')}_{idx}", use_container_width=True):
                    save_feedback_action(user, session_id, event, "want_to_go", rank_position=idx, feedback_reasons=_feedback_reason_value(reason_key))
                    # Want/Maybe should stay visible in Discover. They become playlist badges, not removed cards.
                    st.toast("Saved as Want to Go")
                    st.rerun()
            with action_cols[3]:
                if st.button("Maybe", key=f"maybe_{section}_{session_id}_{event.get('event_id')}_{idx}", use_container_width=True):
                    save_feedback_action(user, session_id, event, "maybe", rank_position=idx, feedback_reasons=_feedback_reason_value(reason_key))
                    # Want/Maybe should stay visible in Discover. They become playlist badges, not removed cards.
                    st.toast("Saved as Maybe")
                    st.rerun()
            with action_cols[4]:
                if st.button("✕ Not a fit", key=f"no_{section}_{session_id}_{event.get('event_id')}_{idx}", use_container_width=True):
                    save_feedback_action(user, session_id, event, "not_for_me", rank_position=idx, feedback_reasons=_feedback_reason_value(reason_key))
                    st.session_state.hidden_event_ids.add(str(event.get("event_id")))
                    st.toast("Hidden as Not a Fit")
                    st.rerun()
            st.markdown('<div class="action-hint">Tickets opens directly. Use Want / Maybe / Not a fit to teach the personal ranker when you train in batches.</div>', unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Find concerts")
    profile_mode = st.radio("Taste profile", ["Spotify login", "Demo profile"], index=0)
    city = st.selectbox("City", CITY_OPTIONS, index=0, accept_new_options=True, help="Start typing to search or add a city.")
    inferred_state = infer_state_for_city(city, "TX")
    state_options = [inferred_state] + [value for value in STATE_OPTIONS if value != inferred_state]
    state = st.selectbox("State", state_options, index=0, accept_new_options=True, help="Start typing to search or add a state code.")
    radius = st.slider("Search radius", 10, 150, 50, step=10, format="%d miles")

    today = date.today()
    date_range = st.date_input("Dates", value=(today, today + timedelta(days=90)))
    if isinstance(date_range, (tuple, list)):
        selected_dates = [value for value in date_range if hasattr(value, "isoformat")]
        start_date = selected_dates[0].isoformat() if selected_dates else today.isoformat()
        end_date = selected_dates[-1].isoformat() if selected_dates else (today + timedelta(days=90)).isoformat()
    elif hasattr(date_range, "isoformat"):
        start_date = end_date = date_range.isoformat()
    else:
        start_date, end_date = today.isoformat(), (today + timedelta(days=90)).isoformat()

    keyword = st.text_input("Artist, genre, or keyword", "", placeholder="country, electronic, Ella Langley...")
    venue_options = AUSTIN_VENUES if city.strip().lower() == "austin" else ["All venues", "Custom venue..."]
    venue_choice = st.selectbox("Venue", venue_options)
    custom_venue = st.text_input("Venue name", "") if venue_choice == "Custom venue..." else ""
    venue_name = "" if venue_choice == "All venues" else (custom_venue.strip() if venue_choice == "Custom venue..." else venue_choice)
    recommendation_mode = st.selectbox(
        "Recommendation style",
        ["Best overall", "Familiar favorites", "Fresh discoveries", "Up & coming"],
        index=0,
    )

    search_speed = st.selectbox(
        "Search speed",
        ["Fast demo", "Balanced", "Deep search"],
        index=0,
        help="Fast demo is best for testing. Deep search finds more shows/prices but is slower.",
    )
    speed_profiles = {
        "Fast demo": {"size": 80, "ticketmaster_pages": 1, "seatgeek_pages": 1, "top_artist_search_count": 10, "price_enrichment_limit": 8, "display_page_size": 20},
        "Balanced": {"size": 120, "ticketmaster_pages": 2, "seatgeek_pages": 2, "top_artist_search_count": 15, "price_enrichment_limit": 15, "display_page_size": 30},
        "Deep search": {"size": 200, "ticketmaster_pages": 5, "seatgeek_pages": 5, "top_artist_search_count": 30, "price_enrichment_limit": 40, "display_page_size": 40},
    }
    speed_config = speed_profiles[search_speed]

    with st.expander("More options"):
        st.caption("Sources, page depth, dedupe, and price enrichment are automatic in V21.")
        group_mode = st.checkbox("Blend a second listener", value=False)
        group_name = "Friend"
        group_artist_text = ""
        if group_mode:
            group_name = st.text_input("Second listener", "Maggie")
            group_artist_text = st.text_area("Their favorite artists", "", height=70)
        refresh_taste = st.checkbox("Refresh Spotify taste", value=False)
        top_artist_search_count = st.slider("Top Spotify artists searched directly", 10, 40, 30, step=5)

    with st.expander("Testing controls"):
        use_feedback_model_toggle = st.checkbox(
            "Use feedback model",
            value=True,
            help="Turn off to test pure Spotify/taste ranking without the 507-rating model.",
        )
        use_saved_history_toggle = st.checkbox(
            "Use saved Want/Maybe history",
            value=True,
            help="Turn off to hide old saved badges, saved-first sorting, hidden history, and My Playlist memory for a clean cold-start test.",
        )
        st.caption("For a clean recommender test, turn both off, then click Get recommendations.")

    # Product defaults: keep technical source choices out of the main UI.
    size = int(speed_config["size"])
    grouping_mode = "Collapse repeated dates"
    search_top_artists = True
    use_ticketmaster = True
    use_seatgeek = True
    use_songkick = False
    ticketmaster_pages = int(speed_config["ticketmaster_pages"])
    seatgeek_pages = int(speed_config["seatgeek_pages"])
    price_enrichment_limit = int(speed_config["price_enrichment_limit"])
    display_page_size = int(speed_config["display_page_size"])
    top_artist_search_count = min(int(top_artist_search_count), int(speed_config["top_artist_search_count"]))

    status = model_status()
    if not use_feedback_model_toggle:
        status = {
            "active": False,
            "label": "Feedback model off",
            "detail": "Test mode: using Spotify/taste ranking only.",
            "bundle": None,
        }
    status_class = "" if status["active"] else " status-base"
    st.markdown(f'<span class="status-pill{status_class}">{status["label"]}</span>', unsafe_allow_html=True)
    st.caption(status["detail"])
    run = st.button("Get recommendations", type="primary", use_container_width=True)


render_hero(city, state, radius, start_date, end_date, keyword)


# ---------------- Recommendation pipeline ----------------
if run:
    session_id = str(uuid.uuid4())
    cache_key = profile_mode
    if refresh_taste:
        st.session_state.taste_cache.pop(cache_key, None)

    with st.spinner("Reading your music taste..."):
        cached = st.session_state.taste_cache.get(cache_key)
        if cached:
            user, primary_top_artists, top_tracks = cached
        elif profile_mode == "Spotify login":
            spotify = get_spotify_client(cache_key=st.session_state.browser_session_id)
            if spotify is None:
                st.error("Spotify is not connected or Streamlit secrets are missing. Use the Connect Spotify button or check SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET / SPOTIFY_REDIRECT_URI.")
                st.stop()
            user = get_current_user(spotify)
            primary_top_artists, top_tracks = get_blended_taste_profile(spotify, artist_limit=50, track_limit=50)
            user["taste_profile_source"] = "Spotify · recent and long-term taste blended"
            st.session_state.taste_cache[cache_key] = (user, primary_top_artists, top_tracks)
        else:
            user, primary_top_artists, top_tracks = demo_profile()
            user["taste_profile_source"] = "Demo taste profile"
            st.session_state.taste_cache[cache_key] = (user, primary_top_artists, top_tracks)

    second_artists = parse_group_artists(group_artist_text) if group_mode else []
    top_artists = primary_top_artists
    if group_mode and second_artists:
        top_artists = add_group_listener_artists(primary_top_artists, group_name, second_artists)
        user = dict(user)
        user["display_name"] = f"{user.get('display_name', 'You')} + {group_name}"

    targeted_artists = []
    if search_top_artists:
        targeted_artists.extend([
            artist.get("artist") for artist in primary_top_artists[:top_artist_search_count]
            if artist.get("artist")
        ])
    for artist_name in second_artists:
        if artist_name not in targeted_artists:
            targeted_artists.append(artist_name)

    with st.spinner("Finding shows and checking live prices..."):
        result = cached_event_search(
            city, state, "US", radius, size, keyword.strip(),
            use_ticketmaster, use_seatgeek, use_songkick,
            ticketmaster_pages, seatgeek_pages,
            start_date, end_date, price_enrichment_limit, top_artist_search_count, venue_name,
            tuple(targeted_artists),
        )

    events = result["events"]
    if not events:
        st.warning("No events were returned. Widen the date range/radius or check source warnings.")
        if result.get("counts", {}).get("errors"):
            st.json(result["counts"]["errors"])
        st.stop()

    current_model = load_feedback_model("current") if use_feedback_model_toggle else None
    use_trained_model = current_model is not None
    with st.spinner("Ranking concerts for your taste..."):
        ranked_raw = rank_events_v6(
            top_artists,
            top_tracks,
            events,
            use_trained_model=use_trained_model,
            recommendation_mode=recommendation_mode,
            model_variant="current",
            model_weight=None,
        )
        ranked_events = collapse_ranked_events(ranked_raw, grouping_mode)
        if group_mode:
            ranked_events = annotate_group_fit(
                ranked_events,
                primary_top_artists,
                second_artists,
                user.get("display_name", "You").split(" + ")[0],
                group_name,
            )

    log_impressions_once(user, session_id, ranked_events, top_n=25)
    st.session_state.recommendation_run = {
        "session_id": session_id,
        "user": user,
        "primary_top_artists": primary_top_artists,
        "group_mode": group_mode,
        "group_name": group_name,
        "group_artists": second_artists,
        "top_artists": top_artists,
        "top_tracks": top_tracks,
        "ranked_events": ranked_events,
        "city": city,
        "state": state,
        "source_counts": result["counts"],
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "venue_name": venue_name,
            "recommendation_mode": recommendation_mode,
            "grouping_mode": grouping_mode,
            "model_active": use_trained_model,
            "model_weight": float(current_model.get("recommended_model_weight") or 0.20) if use_trained_model else 0.0,
            "use_feedback_model": bool(use_feedback_model_toggle),
            "use_saved_history": bool(use_saved_history_toggle),
            "search_speed": search_speed,
            "display_page_size": display_page_size,
            "price_enrichment_limit": price_enrichment_limit,
                "sidebar_keyword": keyword.strip(),
        },
    }
    st.session_state.hidden_event_ids = set()
    st.session_state.copilot_response = None
    st.session_state.night_plan_result = None


if st.session_state.recommendation_run is None:
    st.info("Choose Spotify or Demo, set your city and dates, then click **Get recommendations**.")
    st.stop()

run_data = st.session_state.recommendation_run
session_id = run_data["session_id"]
user = run_data["user"]
top_artists = run_data["top_artists"]
top_tracks = run_data["top_tracks"]
ranked_events = run_data["ranked_events"]
source_counts = run_data["source_counts"]
use_saved_history_toggle = bool(run_data.get("filters", {}).get("use_saved_history", True))
use_feedback_model_toggle = bool(run_data.get("filters", {}).get("use_feedback_model", True))
display_page_size = int(run_data.get("filters", {}).get("display_page_size", 20))
ranked_df = pd.DataFrame(ranked_events)
artists_df = pd.DataFrame(top_artists)
tracks_df = pd.DataFrame(top_tracks)
rated_event_ids = get_user_rated_event_ids(user.get("user_id", "unknown_user")) if use_saved_history_toggle else set()

price_count = sum(
    1 for event in ranked_events
    if event.get("min_price") is not None or event.get("median_price") is not None or event.get("average_price") is not None
)
price_coverage = round(price_count / len(ranked_events) * 100) if ranked_events else 0
direct_count = sum(1 for event in ranked_events if int(event.get("has_direct_artist_match") or 0) == 1)

active_model = run_data.get("filters", {}).get("model_active")
bundle = load_feedback_model("current") if active_model else None
# V29 keeps noisy dashboard stats out of the UI. Keep only the lightweight model badge/status.
ranked_events = _dedupe_events_for_display(ranked_events)
render_stats_strip(len(ranked_events), direct_count, price_coverage, bool(active_model), bundle)

# Playlist memory is shared across Discover, My Playlist, and Copilot unless test mode disables it.
if use_saved_history_toggle:
    playlist_memory_df = load_playlist_preferences(user.get("user_id", "unknown_user"))
else:
    playlist_memory_df = pd.DataFrame()
playlist_memory_events = [] if playlist_memory_df.empty else [interaction_row_to_event(row) for row in playlist_memory_df.to_dict(orient="records")]
active_playlist_events = [] if playlist_memory_df.empty else [interaction_row_to_event(row) for row in playlist_memory_df[playlist_memory_df["action"].isin(["want_to_go", "maybe"])].to_dict(orient="records")]

if source_counts.get("errors"):
    with st.expander("Source warnings"):
        st.json(source_counts.get("errors"))

main_tabs = st.tabs(["Discover", "My Playlist", "Copilot", "Taste by Season", "Behind the Scenes"])




def _dedupe_events_for_display(events):
    """Aggressively merge visually duplicated events across Ticketmaster/SeatGeek."""
    from difflib import SequenceMatcher as _SM
    def clean(v):
        if v is None:
            return ""
        return "".join(ch.lower() for ch in str(v).strip() if ch.isalnum())
    def pick(event, keys):
        if not isinstance(event, dict):
            return ""
        for key in keys:
            val = event.get(key)
            if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
                return val
        raw = event.get("raw_json") or event.get("raw") or {}
        if isinstance(raw, dict):
            for key in keys:
                val = raw.get(key)
                if val is not None and str(val).strip() and str(val).lower() not in ("none", "nan", "tbd"):
                    return val
            dates = raw.get("dates") or {}
            if isinstance(dates, dict):
                start = dates.get("start") or {}
                if isinstance(start, dict):
                    if "date" in " ".join(keys).lower():
                        return start.get("localDate") or start.get("dateTime") or ""
                    if "time" in " ".join(keys).lower():
                        return start.get("localTime") or start.get("dateTime") or ""
        return ""
    def title(event): return str(pick(event, ["event_name", "title", "name"]))
    def venue(event): return str(pick(event, ["venue", "venue_name", "location"]))
    def city(event): return str(pick(event, ["city", "venue_city"]))
    def date_val(event): return str(pick(event, ["event_date", "date", "localDate", "datetime_local", "event_datetime", "start_date", "date_display"]))[:10]
    def time_val(event): return str(pick(event, ["event_time", "time", "localTime", "datetime_local", "event_datetime", "start_time", "time_display"]))[:5]
    def headliner(event):
        artists = event.get("artists") if isinstance(event, dict) else []
        if artists:
            first = str(artists[0] or "").strip()
            if first:
                return clean(first)
        t = str(title(event)).lower()
        t = re.sub(r".*\bpresents\b", " ", t).strip()
        t = re.split(r"\s+(?:with|w/|feat\.?|featuring|and)\s+|\s[-–—:]\s", t)[0]
        t = re.sub(r"\b(tickets|official|live|concert|tour|event|music|festival|weekend|one|two|the)\b", " ", t)
        return clean(t)
    out, seen_exact = [], set()
    for event in events or []:
        if not isinstance(event, dict):
            continue
        d, tm, v, c, h = date_val(event), time_val(event), clean(venue(event)), clean(city(event)), headliner(event)
        t_clean = clean(title(event))
        exact_key = (d, tm, v, c, h or t_clean[:50])
        if exact_key in seen_exact:
            continue
        duplicate = False
        for kept in out:
            kd, ktm, kv, kc, kh = date_val(kept), time_val(kept), clean(venue(kept)), clean(city(kept)), headliner(kept)
            if d != kd:
                continue
            same_place = (v and kv and v == kv) or (c and kc and c == kc)
            same_time = bool(tm and ktm and tm == ktm)
            if not same_place:
                continue
            title_score = _SM(None, clean(title(event)), clean(title(kept))).ratio()
            head_score = _SM(None, h, kh).ratio() if h and kh else 0.0
            if head_score >= 0.72 or title_score >= 0.72 or (same_time and max(head_score, title_score) >= 0.42):
                duplicate = True
                break
        if duplicate:
            continue
        seen_exact.add(exact_key)
        out.append(event)
    return out


# ---------------- Discover ----------------
with main_tabs[0]:
    if use_saved_history_toggle:
        prefs_df, status_by_id, hidden_ids = preference_maps(user.get("user_id", "unknown_user"))
    else:
        prefs_df, status_by_id, hidden_ids = pd.DataFrame(), {}, set()
    base_events = _dedupe_events_for_display([_cc_add_spotify_fields(e) for e in (ranked_events or [])])
    venues = sorted({ _cc_event_venue(e) for e in base_events if _cc_event_venue(e) })
    city_values_current = sorted({ _cc_event_city(e) for e in base_events if _cc_event_city(e) })
    genre_options = ["All genres"] + sorted({str(_cc_genre(e)) for e in base_events if _cc_genre(e)})

    st.markdown(
        f'<div class="discover-title">Personalized concerts for {escape(str(user.get("display_name") or "you"))}</div>'
        '<div class="discover-copy">Use the sidebar to load shows. Then search, filter, save, and plan from the full loaded list.</div>',
        unsafe_allow_html=True,
    )

    f_search, f_city, f_venue, f_genre, f_hidden = st.columns([2.35, .95, 1.15, 1.05, .65], vertical_alignment="bottom")
    with f_search:
        discover_search_text = st.text_input("Search artist, show, genre, venue", placeholder="Type Ella, ACL, electronic, Moody Center...", key="discover_search_text_v40")
    discover_sort = "Saved first"  # V41: no Sort dropdown; automatic saved-first ranking
    with f_city:
        discover_city = st.selectbox("City", ["All cities"] + city_values_current, key="discover_city_filter_v40")
    with f_venue:
        display_venue = st.selectbox("Venue", ["All venues"] + venues, key="discover_venue_filter_v40")
    with f_genre:
        discover_genre = st.selectbox("Genre", genre_options, key="discover_genre_v40")
    with f_hidden:
        show_hidden = st.checkbox("Hidden", value=False, help="Show events marked Not a Fit.", key="discover_hidden_v40")

    visible = _cc_apply_discover_filters(
        base_events,
        query=discover_search_text,
        genre=discover_genre,
        match_type=None,
        sort_mode=None,
        city_filter=discover_city,
        venue_filter=display_venue,
    )
    if not show_hidden:
        visible = [event for event in visible if str(event.get("event_id")) not in hidden_ids and str(event.get("event_id")) not in st.session_state.hidden_event_ids]

    if use_saved_history_toggle:
        # Normal product mode: saved Want first, saved Maybe second, then Match Score.
        visible = _cc_sort_saved_first(visible, status_by_id)
        caption = f"Showing {len(visible)} of {len(base_events)} loaded shows. Saved Want and Maybe stay first, then new shows rank by Match Score."
    else:
        # Clean test mode: ignore saved-first memory and rank by current recommender score only.
        visible = sorted(visible, key=lambda e: _cc_event_score(e), reverse=True)
        caption = f"Clean test: showing {len(visible)} of {len(base_events)} loaded shows. Saved history is ignored; current ranking score only."

    visible = _dedupe_events_for_display(visible)
    st.caption(caption)
    visible = _cc_limit_visible_events(visible, key=f"discover_{session_id}", page_size=display_page_size)

    if not visible:
        st.info("No concerts match the current filters. Clear search, widen filters, or turn on Hidden.")
    for index, event in enumerate(visible, 1):
        saved_badge = preference_label(status_by_id.get(str(event.get("event_id")))) if use_saved_history_toggle else None
        extra = saved_badge or event.get("group_fit_label")
        render_event_card(event, index, "feed", user, session_id, extra)


# ---------------- Playlist ----------------
with main_tabs[1]:
    st.markdown('<div class="section-head">My Playlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Every show you have acted on. Want and Maybe are saved options. Not a Fit is hidden from Discover unless you turn it back on. V36 keeps your trained model and attempts to restore old saved actions from prior local folders.</div>', unsafe_allow_html=True)
    playlist = load_playlist_preferences(user.get("user_id", "unknown_user")) if use_saved_history_toggle else pd.DataFrame()

    if not use_saved_history_toggle:
        st.info("Saved history is OFF for this test run. Turn it back on in Testing controls to view My Playlist.")
    elif playlist.empty:
        st.info("No saved playlist rows are showing yet. Your trained model is still active, but saved Want/Maybe UI rows may need a database restore. Use Want, Maybe, or Not a Fit on Discover to rebuild this list.")
    else:
        want_count = int((playlist["action"] == "want_to_go").sum())
        maybe_count = int((playlist["action"] == "maybe").sum())
        no_count = int((playlist["action"] == "not_for_me").sum())
        active_saved_count = want_count + maybe_count
        city_values = sorted([v for v in playlist.get("city", pd.Series(dtype=str)).dropna().unique().tolist() if v])
        venue_values = sorted([v for v in playlist.get("venue", pd.Series(dtype=str)).dropna().unique().tolist() if v])

        st.markdown(
            f"""<div class="shortlist-summary">
                <div class="shortlist-summary-card"><div class="num">{len(playlist)}</div><div class="lbl">All shows</div></div>
                <div class="shortlist-summary-card"><div class="num">{want_count}</div><div class="lbl">Want</div></div>
                <div class="shortlist-summary-card"><div class="num">{maybe_count}</div><div class="lbl">Maybe</div></div>
                <div class="shortlist-summary-card"><div class="num">{no_count}</div><div class="lbl">Don't go</div></div>
              </div>""",
            unsafe_allow_html=True,
        )

        f1, f2, f3, f4 = st.columns([1.35, 1.05, 1.2, 1.25])
        with f1:
            category = st.radio("Show", ["All shows", "Want", "Maybe", "Don't go"], horizontal=True)
        with f2:
            playlist_city = st.selectbox("City", ["All cities"] + city_values, key="playlist_city")
        with f3:
            playlist_venue = st.selectbox("Venue", ["All venues"] + venue_values, key="playlist_venue")
        with f4:
            sort_choice = st.selectbox("Sort", ["Soonest", "Best match", "Lowest price", "Recently saved"])

        filtered_playlist = playlist.copy()
        if category == "Want":
            filtered_playlist = filtered_playlist[filtered_playlist["action"] == "want_to_go"]
        elif category == "Maybe":
            filtered_playlist = filtered_playlist[filtered_playlist["action"] == "maybe"]
        elif category == "Don't go":
            filtered_playlist = filtered_playlist[filtered_playlist["action"] == "not_for_me"]
        if playlist_city != "All cities":
            filtered_playlist = filtered_playlist[filtered_playlist["city"] == playlist_city]
        if playlist_venue != "All venues":
            filtered_playlist = filtered_playlist[filtered_playlist["venue"] == playlist_venue]
        if sort_choice == "Soonest" and "event_date" in filtered_playlist.columns:
            filtered_playlist = filtered_playlist.sort_values(["event_date", "event_time"], ascending=True)
        elif sort_choice == "Best match" and "final_score" in filtered_playlist.columns:
            filtered_playlist = filtered_playlist.sort_values("final_score", ascending=False)
        elif sort_choice == "Lowest price" and "min_price" in filtered_playlist.columns:
            filtered_playlist = filtered_playlist.assign(_price_sort=filtered_playlist["min_price"].fillna(999999)).sort_values("_price_sort", ascending=True).drop(columns=["_price_sort"])
        elif "created_at" in filtered_playlist.columns:
            filtered_playlist = filtered_playlist.sort_values("created_at", ascending=False)

        st.markdown('<div class="copilot-note"><b>Product logic:</b> Want/Maybe stay visible on Discover and collect here. Don\'t Go is hidden from Discover by default but can be reviewed or restored here.</div>', unsafe_allow_html=True)

        if filtered_playlist.empty:
            st.info("No shows match those filters.")
        for index, row in enumerate(filtered_playlist.to_dict(orient="records"), 1):
            event = _cc_add_spotify_fields(interaction_row_to_event(row))
            action = row.get("action")
            if action == "want_to_go":
                status_label = "Want"
                status_class = "badge-direct"
            elif action == "maybe":
                status_label = "Maybe"
                status_class = "badge-group"
            else:
                status_label = "Don't go"
                status_class = "badge-warn"
            links = event_links(event)
            title = escape(str(event.get("event_name") or "Untitled event"))
            venue_line = escape(f"{event.get('venue') or 'Venue TBD'} · {event.get('city') or ''}, {event.get('state') or ''}".strip(" ·,"))
            when_line = escape(format_when(event))
            reason = escape(str(event.get("why_recommended") or event.get("why_artist_match") or "Saved from your recommendation list."))
            price_text, price_class = price_label(event)
            col_main, col_img = st.columns([4.5, 1.25], vertical_alignment="top")
            with col_main:
                st.markdown(
                    f"""<div class="shortlist-card">
                        <div class="shortlist-title">{title}</div>
                        <div class="shortlist-meta"><b>{when_line}</b> · {venue_line}</div>
                        <span class="badge badge-price">{escape(_cc_match_score_label(event))}</span>
                        <span class="badge {status_class}">{status_label}</span>
                        <span class="badge {price_class}">{escape(str(price_text))}</span>
                        <span class="badge">{escape(str(event.get('winning_genre_cluster_label') or event.get('genre') or 'Music'))}</span>
                        <div class="shortlist-reason">{reason}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                controls = st.columns([1.0, 1.05, .85, .85, 1.0, 1.0])
                with controls[0]:
                    if links:
                        st.link_button("Tickets", links[0], use_container_width=True)
                        # V36 HOTFIX fallback Spotify button
                        try:
                            st.link_button(event.get('spotify_label') or _cc_spotify_label(event), event.get('spotify_url') or _cc_spotify_url(event))
                        except Exception:
                            pass
                with controls[1]:
                    if st.button("Plan night", key=f"playlist_plan_{event.get('event_id')}_{index}", use_container_width=True):
                        st.session_state.plan_event_id = event.get("event_id")
                        st.info("Saved this show for Plan a Night. Open Copilot → Plan a Night and it will be selected.")
                with controls[2]:
                    if action != "want_to_go" and st.button("Want", key=f"playlist_want_{event.get('event_id')}_{index}", use_container_width=True):
                        save_feedback_action(user, session_id, event, "want_to_go", feedback_reasons=["Playlist update"])
                        st.rerun()
                with controls[3]:
                    if action != "maybe" and st.button("Maybe", key=f"playlist_maybe_{event.get('event_id')}_{index}", use_container_width=True):
                        save_feedback_action(user, session_id, event, "maybe", feedback_reasons=["Playlist update"])
                        st.rerun()
                with controls[4]:
                    if action != "not_for_me" and st.button("Don't go", key=f"playlist_no_{event.get('event_id')}_{index}", use_container_width=True):
                        save_feedback_action(user, session_id, event, "not_for_me", feedback_reasons=["Playlist hidden"])
                        st.session_state.hidden_event_ids.add(str(event.get("event_id")))
                        st.rerun()
                with controls[5]:
                    if st.button("Clear", key=f"playlist_clear_{event.get('event_id')}_{index}", use_container_width=True):
                        clear_feedback_preference(user.get("user_id", "unknown_user"), event.get("event_id"))
                        st.session_state.hidden_event_ids.discard(str(event.get("event_id")))
                        st.rerun()
            with col_img:
                if event.get("image_url"):
                    st.image(event.get("image_url"), use_container_width=True)

# ---------------- Copilot ----------------
with main_tabs[2]:
    st.markdown('<div class="section-head">Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask for the best shows or instantly turn one concert into a full night out. Fast mode answers from the ranked event context instantly; Full AI adds place enrichment and a longer generated report.</div>', unsafe_allow_html=True)

    copilot_speed_mode = st.radio(
        "Copilot speed",
        ["Fast answer", "Full AI report"],
        horizontal=True,
        index=0,
        help="Fast answer skips slow place/weather enrichment and the long narrative LLM call. Full AI is slower but richer.",
        key="copilot_speed_mode_v1",
    )
    copilot_fast_mode = copilot_speed_mode == "Fast answer"
    if copilot_fast_mode:
        st.caption("Fast mode: instant deterministic picks from your ranked shows. No slow places/weather scan unless you plan a specific night.")
    else:
        st.caption("Full AI mode: enriches top events with place context and generates a longer Copilot report.")

    _saved_events = active_playlist_events if 'active_playlist_events' in globals() else []
    _playlist_events = playlist_memory_events if 'playlist_memory_events' in globals() else []
    _current_events = ranked_events if 'ranked_events' in globals() else []
    _all_memory_events = unique_events_by_id((_saved_events or []) + (_current_events or [])) if 'unique_events_by_id' in globals() else ((_saved_events or []) + (_current_events or []))

    def _is_good_value(v):
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip() not in ('', 'None', 'nan', 'NaN', 'Date TBD', 'Time TBD', 'Date TBD · Time TBD')
        return True

    def _event_title(event):
        event = event or {}
        return str(event.get('event_name') or event.get('title') or event.get('name') or 'Untitled show')

    def _event_venue(event):
        event = event or {}
        return str(event.get('venue') or event.get('venue_name') or event.get('location') or 'Venue TBD')

    def _event_city(event):
        event = event or {}
        return str(event.get('city') or event.get('venue_city') or '').strip()

    def _event_state(event):
        event = event or {}
        return str(event.get('state') or event.get('venue_state') or '').strip()

    def _event_key(event):
        event = event or {}
        eid = event.get('event_id') or event.get('external_event_id') or event.get('id')
        if eid:
            return ('id', str(eid))
        return ('namevenue', (_event_title(event).lower(), _event_venue(event).lower()))

    # Build a full-event lookup from the original ranked and playlist rows. This fixes the TBD problem:
    # Copilot contexts sometimes carry a trimmed event dict, while Discover has the real date/time.
    def _norm_key_text(v):
        if v is None:
            return ""
        return "".join(ch.lower() for ch in str(v).strip() if ch.isalnum())

    _full_event_index = {}
    for _source_event in ((_current_events or []) + (_saved_events or []) + (_playlist_events or [])):
        if not isinstance(_source_event, dict):
            continue
        _full_event_index[_event_key(_source_event)] = _source_event
        _title_key = ('title', _event_title(_source_event).lower())
        if _title_key not in _full_event_index:
            _full_event_index[_title_key] = _source_event
        _norm_title_key = ('norm_title', _norm_key_text(_event_title(_source_event)))
        if _norm_title_key not in _full_event_index:
            _full_event_index[_norm_title_key] = _source_event
        _norm_title_venue_key = ('norm_title_venue', _norm_key_text(_event_title(_source_event)), _norm_key_text(_event_venue(_source_event)))
        if _norm_title_venue_key not in _full_event_index:
            _full_event_index[_norm_title_venue_key] = _source_event

    def _hydrate_event(event):
        if not isinstance(event, dict):
            return {}
        full = (_full_event_index.get(_event_key(event)) or _full_event_index.get(('title', _event_title(event).lower())) or _full_event_index.get(('norm_title', _norm_key_text(_event_title(event)))) or _full_event_index.get(('norm_title_venue', _norm_key_text(_event_title(event)), _norm_key_text(_event_venue(event)))) or {})
        merged = dict(full) if isinstance(full, dict) else {}
        # Add context event values, but do not let TBD/blank values overwrite real fields.
        for k, v in event.items():
            if _is_good_value(v):
                merged[k] = v
            elif k not in merged:
                merged[k] = v
        return merged

    def _nested_get(d, path):
        cur = d
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    def _first_value(event, keys):
        event = _hydrate_event(event or {})
        for key in keys:
            val = event.get(key)
            if _is_good_value(val):
                return val
        raw = event.get('raw_json') or event.get('raw') or event.get('source_json') or {}
        if isinstance(raw, dict):
            for key in keys:
                val = raw.get(key)
                if _is_good_value(val):
                    return val
            for path in [
                ['dates', 'start', 'localDate'], ['dates', 'start', 'dateTime'], ['dates', 'start', 'localTime'],
                ['datetime_local'], ['datetime_utc'], ['startdatetime'], ['start', 'datetime'], ['event', 'dates', 'start', 'localDate'], ['event', 'dates', 'start', 'localTime']
            ]:
                val = _nested_get(raw, path if isinstance(path, list) else [path])
                if _is_good_value(val):
                    # Only return the first useful raw value if it matches requested key type or is an ISO datetime.
                    key_text = ' '.join(keys).lower()
                    val_text = str(val)
                    if ('date' in key_text and ('-' in val_text or '/' in val_text or 'T' in val_text)) or ('time' in key_text and (':' in val_text or 'T' in val_text)) or ('datetime' in key_text):
                        return val
        return None

    def _pretty_when(event):
        event = _hydrate_event(event or {})
        try:
            existing = format_when(event) if 'format_when' in globals() else ''
            if existing and 'TBD' not in str(existing) and str(existing).strip():
                return str(existing)
        except Exception:
            pass
        date_val = _first_value(event, ['event_date', 'date', 'local_date', 'localDate', 'start_date', 'startDate', 'event_start_date', 'event_datetime', 'datetime', 'datetime_local', 'date_display', 'display_date', 'formatted_date', 'eventDate', 'start_local_date', 'date_local', 'parsed_date', 'date_start', 'start_local', 'show_date'])
        time_val = _first_value(event, ['event_time', 'time', 'local_time', 'localTime', 'start_time', 'startTime', 'event_start_time', 'event_datetime', 'datetime', 'datetime_local', 'time_display', 'display_time', 'formatted_time', 'eventTime', 'start_local_time', 'time_local', 'parsed_time', 'time_start', 'show_time'])
        # Pull date/time out of ISO-ish datetime if needed.
        if date_val and 'T' in str(date_val):
            parts = str(date_val).split('T')
            date_val = parts[0]
            if not time_val and len(parts) > 1:
                time_val = parts[1][:5]
        if time_val and 'T' in str(time_val):
            time_val = str(time_val).split('T')[-1][:5]
        # If the date is yyyy-mm-dd, make it easier to read without extra deps.
        date_txt = str(date_val) if _is_good_value(date_val) else 'Date TBD'
        time_txt = str(time_val)[:5] if _is_good_value(time_val) else 'Time TBD'
        return f"{date_txt} · {time_txt}"

    def _extract_price(event):
        event = _hydrate_event(event or {})
        for key in ['min_price', 'price_min', 'lowest_price', 'ticket_min', 'ticket_price_min', 'price', 'average_price', 'avg_price', 'max_price', 'price_max']:
            val = event.get(key)
            if _is_good_value(val):
                try:
                    return f"From ${float(val):.0f}"
                except Exception:
                    return f"Price {val}"
        raw = event.get('raw_json') or event.get('raw') or {}
        if isinstance(raw, dict):
            ranges = raw.get('priceRanges') or raw.get('price_ranges') or []
            if isinstance(ranges, list) and ranges:
                vals = []
                for r in ranges:
                    if isinstance(r, dict):
                        for k in ['min', 'minimum', 'price_min']:
                            if _is_good_value(r.get(k)):
                                vals.append(r.get(k))
                if vals:
                    try:
                        return f"From ${float(min(vals)):.0f}"
                    except Exception:
                        return f"From ${vals[0]}"
            stats = raw.get('stats') or {}
            if isinstance(stats, dict):
                for key in ['lowest_price', 'average_price', 'median_price', 'highest_price']:
                    val = stats.get(key)
                    if _is_good_value(val):
                        try:
                            return f"From ${float(val):.0f}"
                        except Exception:
                            return f"Price {val}"
        return "Price not returned"

    def _ticket_url(event):
        event = _hydrate_event(event or {})
        for key in ['ticket_url', 'url', 'event_url', 'tickets_url', 'seatgeek_url', 'ticketmaster_url']:
            val = event.get(key)
            if val:
                return val
        raw = event.get('raw_json') or event.get('raw') or {}
        if isinstance(raw, dict):
            for key in ['url', 'ticket_url', 'event_url']:
                if raw.get(key):
                    return raw.get(key)
        return None

    def _city_options_for(events):
        vals = []
        for e in events or []:
            e = _hydrate_event(e)
            c = _event_city(e)
            stt = _event_state(e)
            if c:
                vals.append(f"{c}, {stt}" if stt else c)
        vals = sorted(set(vals))
        sidebar_city = str(st.session_state.get('city') or run_data.get('filters', {}).get('city') or '').strip() if 'run_data' in globals() else ''
        sidebar_state = str(st.session_state.get('state') or run_data.get('filters', {}).get('state') or '').strip() if 'run_data' in globals() else ''
        preferred = f"{sidebar_city}, {sidebar_state}" if sidebar_city and sidebar_state else sidebar_city
        opts = ["All cities"] + vals
        if preferred and preferred in opts:
            opts.remove(preferred)
            opts.insert(1, preferred)
        return opts

    def _filter_city(events, label):
        if not events:
            return []
        if not label or label == "All cities":
            return [_hydrate_event(e) for e in events]
        city = label.split(',')[0].strip().lower()
        return [_hydrate_event(e) for e in events if _event_city(_hydrate_event(e)).lower() == city]

    def _safe_score(context, label):
        scores = context.get('copilot_scores', {}) or {} if isinstance(context, dict) else {}
        key_map = {
            'Recommended': 'request_match',
            'Top artist match': 'top_artist',
            'Most compatible': 'compatible',
            'New discovery': 'discovery',
            'Best night out': 'night_out',
        }
        return scores.get(key_map.get(label, 'overall'), scores.get('overall', 0))

    def _set_plan_event(event, source='Copilot picks'):
        event = _hydrate_event(event or {})
        st.session_state.plan_event_id = event.get('event_id')
        st.session_state.plan_event_name = _event_title(event)
        st.session_state.plan_event_city = _event_city(event)
        st.session_state.plan_event_payload = event
        st.session_state.plan_source_request = source
        return event

    def _normalize_places(items):
        out = []
        def add(x):
            if isinstance(x, dict):
                out.append(x)
            elif isinstance(x, (list, tuple)):
                for y in x:
                    add(y)
        add(items or [])
        return out

    def _places_from_plan(structured):
        if not isinstance(structured, dict):
            return []
        for key in ['nearby_options', 'nearby_places', 'places', 'place_options', 'recommendations']:
            vals = _normalize_places(structured.get(key))
            if vals:
                return vals
        timeline = _normalize_places(structured.get('timeline') or structured.get('steps'))
        vals = []
        for item in timeline:
            place = item.get('place') or item.get('venue') or item.get('restaurant') or item.get('bar')
            if isinstance(place, dict):
                vals.append(place)
            elif isinstance(place, str) and place.strip():
                vals.append({'name': place})
        return vals

    def _call_structured_plan(event, night_style, notes):
        event = _hydrate_event(event or {})
        attempts = [
            lambda: build_structured_night_plan(event, night_style=night_style, notes=notes, radius_miles=None, place_focus='auto'),
            lambda: build_structured_night_plan(event, night_style=night_style, notes=notes),
            lambda: build_structured_night_plan(event, night_style, notes),
            lambda: build_structured_night_plan(event),
        ]
        for attempt in attempts:
            try:
                return attempt()
            except TypeError:
                continue
            except Exception as e:
                st.session_state.last_plan_error = str(e)
                return None
        return None

    def _call_llm_plan(event, night_style, notes):
        event = _hydrate_event(event or {})
        attempts = [
            lambda: create_night_plan(event, night_style=night_style, notes=notes, radius_miles=None, place_focus='auto'),
            lambda: create_night_plan(event, night_style=night_style, notes=notes),
            lambda: create_night_plan(event, night_style, notes),
        ]
        for attempt in attempts:
            try:
                return attempt()
            except TypeError:
                continue
            except Exception as e:
                st.session_state.last_llm_plan_error = str(e)
                return None
        return None

    def _fallback_structured_plan(event, structured=None):
        event = _hydrate_event(event or {})
        places = _places_from_plan(structured)
        before = places[0].get('name') if places else 'Dinner or drinks nearby'
        after = places[1].get('name') if len(places) > 1 else 'Easy nearby after-show option'
        venue = _event_venue(event)
        title = _event_title(event)
        when = _pretty_when(event)
        show_time = 'Show time'
        if '·' in when and 'TBD' not in when:
            show_time = when.split('·')[-1].strip()
        return {
            'timeline': [
                {'time': '2 hours before', 'title': 'Pre-show', 'place': before, 'description': 'Start with food or drinks close enough that the night stays easy.'},
                {'time': '45 minutes before', 'title': 'Head to venue', 'place': venue, 'description': 'Build in a buffer so you are not rushed.'},
                {'time': show_time, 'title': 'Show', 'place': title, 'description': 'Main event.'},
                {'time': 'After show', 'title': 'After-show option', 'place': after, 'description': 'Keep a flexible backup for drinks, dessert, or heading home.'},
            ],
            'nearby_options': places,
        }

    def _timeline_has_content(structured):
        if not isinstance(structured, dict):
            return False
        timeline = structured.get('timeline') or structured.get('steps') or []
        if not isinstance(timeline, list) or not timeline:
            return False
        for item in timeline:
            if isinstance(item, dict):
                vals = [item.get(k) for k in ['title', 'label', 'activity', 'place', 'description', 'details', 'notes']]
                if any(_is_good_value(v) for v in vals):
                    return True
        return False

    def _auto_plan(event, style='date night', notes='Keep it easy. Good food or drinks nearby, no rushing, and a backup option.', source='Copilot picks'):
        event = _set_plan_event(event, source=source)
        structured = _call_structured_plan(event, style, notes)
        llm_plan = None if copilot_fast_mode else _call_llm_plan(event, style, notes)
        if not _timeline_has_content(structured):
            structured = _fallback_structured_plan(event, structured)
        st.session_state.simple_night_plan = {'event': event, 'structured': structured, 'llm': llm_plan, 'style': style, 'notes': notes}
        return structured, llm_plan

    def _plain_event_card(event, label, score=None, button_key_suffix=''):
        event = _hydrate_event(event or {})
        title = _event_title(event)
        venue = _event_venue(event)
        city = _event_city(event)
        state = _event_state(event)
        when = _pretty_when(event)
        price = _extract_price(event)
        why = str(event.get('why_recommended') or event.get('why_artist_match') or event.get('why_it_fits') or 'Strong fit based on your taste, ranking signals, and event context.')
        score_line = f"<div class='mini-score'>Copilot score {score:.2f}</div>" if isinstance(score, (int, float)) else ""
        st.markdown(f"""
        <div class="shortlist-card">
          <div class="badge badge-direct">{label}</div>
          <div class="shortlist-title">{title}</div>
          <div class="shortlist-meta"><b>{when}</b> · {venue} · {city}, {state}</div>
          <span class="badge">{price}</span>
          <div class="shortlist-reason">{why}</div>
          {score_line}
        </div>
        """, unsafe_allow_html=True)
        url = _ticket_url(event)
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button('Plan this night', key=f"auto_plan_{button_key_suffix}_{label}_{event.get('event_id') or title}", use_container_width=True):
                with st.spinner('Building fast plan...'):
                    _auto_plan(event, source='Copilot picks')
                st.success('Night planned below. You can also open Plan a Night to edit it.')
        with b2:
            if url:
                st.link_button('Tickets', url, use_container_width=True)
                # V36 HOTFIX fallback Spotify button
                try:
                    st.link_button(event.get('spotify_label') or _cc_spotify_label(event), event.get('spotify_url') or _cc_spotify_url(event))
                except Exception:
                    pass

    def _item_text(item, keys, default=''):
        if isinstance(item, dict):
            for k in keys:
                v = item.get(k)
                if _is_good_value(v):
                    return str(v)
        return default

    def _render_plan(plan):
        if not plan:
            return
        event = _hydrate_event(plan.get('event') or {})
        st.markdown('#### Your Night Out')
        st.markdown(f"**{_event_title(event)}**  \n{_pretty_when(event)} · {_event_venue(event)} · {_event_city(event)}, {_event_state(event)}")
        structured = plan.get('structured')
        if structured and isinstance(structured, dict):
            timeline = structured.get('timeline') or structured.get('steps') or []
            timeline = _normalize_places(timeline)
            if timeline:
                cols = st.columns(min(4, len(timeline)))
                for i, item in enumerate(timeline[:4]):
                    with cols[i % len(cols)]:
                        t = _item_text(item, ['time', 'start_time', 'when'], f'Step {i+1}')
                        title = _item_text(item, ['title', 'label', 'activity', 'headline', 'step'], ['Pre-show', 'Head to venue', 'Show', 'After show'][i] if i < 4 else 'Step')
                        place = _item_text(item, ['place', 'venue', 'restaurant', 'bar', 'name', 'detail', 'description', 'details', 'notes'], 'Details loading')
                        st.markdown(f"<div class='mini-card'><b>{t}</b><br><strong>{title}</strong><br><span>{place}</span></div>", unsafe_allow_html=True)
            nearby = _places_from_plan(structured)
            if nearby:
                st.markdown('#### Nearby options')
                cards = st.columns(min(3, len(nearby)))
                for i, place in enumerate(nearby[:9]):
                    with cards[i % len(cards)]:
                        name = place.get('name') or place.get('title') or 'Nearby option'
                        rating = place.get('rating')
                        dist = place.get('distance_miles') or place.get('distance')
                        category = place.get('category') or place.get('type') or place.get('categories') or ''
                        addr = place.get('address') or ''
                        url = place.get('url') or place.get('maps_url')
                        st.markdown(f"<div class='mini-card'><strong>{name}</strong><br><span>{category}</span><br><span>{'' if dist is None else str(dist)+' mi'} {'' if rating is None else '· '+str(rating)+' ★'}</span><br><span>{addr}</span></div>", unsafe_allow_html=True)
                        if url:
                            st.link_button('View place', url, use_container_width=True)
            summary = structured.get('full_plan') or structured.get('summary')
            if summary:
                with st.expander("Copilot's full plan", expanded=True):
                    st.markdown(summary)
        llm_plan = plan.get('llm')
        if llm_plan:
            try:
                render_llm_result(llm_plan, fallback_title='The plan')
            except Exception:
                text = getattr(llm_plan, 'text', None) or getattr(llm_plan, 'content', None) or str(llm_plan)
                st.markdown(text)
        if not structured and not llm_plan:
            err = st.session_state.get('last_plan_error') or st.session_state.get('last_llm_plan_error')
            if err:
                st.warning(f"Plan generator returned no plan. Last error: {err}")
            else:
                st.info('Plan generator returned no plan. Try another show or check place API keys.')

    copilot_tabs = st.tabs(["Ask Copilot", "Plan a Night"])

    with copilot_tabs[0]:
        st.markdown('### Ask Copilot')
        st.markdown('Pick a city, tell Copilot what you want, and it will return a clean set of options. City is a hard filter before recommendations are generated.')

        c1, c2 = st.columns([1.05, 1.3], vertical_alignment='bottom')
        city_opts = _city_options_for(_all_memory_events)
        with c1:
            selected_city = st.selectbox('City', city_opts, index=1 if len(city_opts) > 1 else 0, key='simple_copilot_city')
        with c2:
            source_choice = st.radio('Use', ['Both', 'Current search', 'My Playlist'], horizontal=True, index=0, key='simple_copilot_source')

        quick = st.radio('Quick goal', ['Best overall', 'Date night', 'Top artists', 'New discovery', 'Weekend plan'], horizontal=True, index=0, key='simple_goal_chip')
        goal_defaults = {
            'Best overall': 'Recommend the best shows in this city. Separate recommended, top artist match, most compatible, new discovery, and best night-out plan.',
            'Date night': 'Find the best date-night concerts in this city. Favor reliable venues, good timing, and nearby dinner or drinks.',
            'Top artists': 'Find the strongest direct or near-direct artist matches in this city and explain why they fit my taste.',
            'New discovery': 'Find a fresh discovery in this city that still fits my taste. Avoid obvious picks unless they are clearly best.',
            'Weekend plan': 'Find the best weekend show in this city and explain the best full night-out option.',
        }
        user_question = st.text_area('What are you looking for?', value=goal_defaults.get(quick, goal_defaults['Best overall']), height=110, key='simple_copilot_prompt')

        if source_choice == 'Current search':
            base_events = _current_events
        elif source_choice == 'My Playlist':
            base_events = _saved_events or _playlist_events
        else:
            base_events = _all_memory_events
        city_events = _filter_city(base_events, selected_city)
        st.caption(f"Using {len(city_events)} shows after city/source filtering. Fast mode uses model scores, dates, venues, prices when returned, and taste signals without slow place/weather scans.")

        if st.button('Find my best shows', type='primary', use_container_width=True, key='find_simple_copilot'):
            if not city_events:
                st.warning('No shows match that city/source yet. Run a search for that city or switch source to Both/Current search.')
            else:
                with st.spinner('Finding your best options...' if copilot_fast_mode else 'Building full AI report with places and rankings...'):
                    try:
                        filtered_events, requested_window = _filter_by_request_window(city_events, user_question)
                    except Exception:
                        filtered_events, requested_window = city_events, None
                    candidate_events = filtered_events if requested_window and filtered_events else city_events
                    # Speed: Copilot only needs the best candidates, not the whole event pool.
                    candidate_limit = 35 if copilot_fast_mode else 80
                    enrich_top_n = 0 if copilot_fast_mode else min(8, len(candidate_events))
                    candidate_events = [_hydrate_event(e) for e in candidate_events[:candidate_limit]]
                    try:
                        contexts = enrich_candidate_context(
                            candidate_events,
                            vibe='auto',
                            radius_miles=1.1,
                            enrich_top_n=enrich_top_n,
                            use_places=not copilot_fast_mode,
                            use_weather=False,
                        )
                    except Exception:
                        contexts = [{'event': e, 'event_id': e.get('event_id'), 'copilot_scores': {'overall': e.get('final_score', 0), 'request_match': e.get('final_score', 0)}} for e in candidate_events[:candidate_limit]]
                    # Hydrate context events again after enrichment so card dates/times come from the full recommendation row.
                    for ctx in contexts:
                        if isinstance(ctx, dict):
                            ctx['event'] = _hydrate_event(ctx.get('event') or ctx)
                    try:
                        picks = select_copilot_picks(contexts, situation='auto', user_goal=user_question)
                    except Exception:
                        labels = ['Recommended', 'Top artist match', 'Most compatible', 'New discovery', 'Best night out']
                        picks = {lab: (contexts[i] if i < len(contexts) else None) for i, lab in enumerate(labels)}
                    # Hydrate picks too.
                    for lab, ctx in list((picks or {}).items()):
                        if isinstance(ctx, dict):
                            ctx['event'] = _hydrate_event(ctx.get('event') or ctx)
                    if copilot_fast_mode:
                        narrative = None
                    else:
                        try:
                            narrative = generate_elite_copilot_report(user_question, contexts, picks, top_artists, top_tracks, situation='auto', budget=None)
                        except Exception:
                            narrative = None
                    st.session_state.simple_copilot_result = {'city': selected_city, 'source': source_choice, 'question': user_question, 'requested_window': requested_window, 'picks': picks, 'narrative': narrative}

        result = st.session_state.get('simple_copilot_result')
        if result:
            req = result.get('requested_window')
            if req:
                if req.get('matched_count', 0) > 0:
                    st.info(f"Date window applied: {req.get('label')} ({req.get('start')} to {req.get('end')}).")
                else:
                    st.warning(f"Date window detected: {req.get('label')}, but nothing in the retrieved list matched it. Expand sidebar dates or search again.")
            picks = result.get('picks') or {}
            cols = st.columns(min(5, max(1, len(picks))))
            for idx, (label, ctx) in enumerate(picks.items()):
                if not ctx:
                    continue
                event = _hydrate_event(ctx.get('event', ctx) if isinstance(ctx, dict) else ctx)
                score = _safe_score(ctx, label) if isinstance(ctx, dict) else (event or {}).get('final_score')
                with cols[idx % len(cols)]:
                    _plain_event_card(event, label, score, button_key_suffix='ask')
            if st.session_state.get('simple_night_plan'):
                st.markdown('---')
                _render_plan(st.session_state.get('simple_night_plan'))
            narrative = result.get('narrative')
            if not narrative:
                st.caption("Fast Copilot answer: picks are generated from the ranked event context without waiting on the long AI report.")
            if narrative:
                try:
                    render_llm_result(narrative, fallback_title='Copilot recommendation')
                except Exception:
                    text = getattr(narrative, 'text', None) or getattr(narrative, 'content', None) or str(narrative)
                    st.markdown(text)

    with copilot_tabs[1]:
        st.markdown('### Plan a Night')
        st.markdown('Pick a concert and describe the vibe. Copilot automatically uses the concert city, venue, show time, nearby places, ratings, prices when returned, and your notes.')
        source_for_plan = st.radio('Choose from', ['Recommended list', 'My Playlist', 'Copilot picks'], horizontal=True, index=['Recommended list','My Playlist','Copilot picks'].index(st.session_state.get('plan_source_request', 'Recommended list')) if st.session_state.get('plan_source_request', 'Recommended list') in ['Recommended list','My Playlist','Copilot picks'] else 0, key='plan_source_simple_widget')
        if source_for_plan == 'My Playlist':
            plan_pool = _saved_events or _playlist_events
        elif source_for_plan == 'Copilot picks' and st.session_state.get('simple_copilot_result'):
            plan_pool = []
            for ctx in (st.session_state.simple_copilot_result.get('picks') or {}).values():
                if ctx:
                    plan_pool.append(_hydrate_event(ctx.get('event', ctx) if isinstance(ctx, dict) else ctx))
        else:
            plan_pool = _current_events
        plan_pool = [_hydrate_event(e) for e in (plan_pool or [])]
        if not plan_pool:
            st.info('No concerts available in this source yet. Run a search, save shows, or use Ask Copilot first.')
        else:
            plan_option_limit = 50 if copilot_fast_mode else 100
            options = [f"{i+1}. {_event_title(e)} — {_pretty_when(e)} — {_event_city(e)}" for i, e in enumerate(plan_pool[:plan_option_limit])]
            default_index = 0
            remembered_id = st.session_state.get('plan_event_id')
            if remembered_id:
                for i, e in enumerate(plan_pool[:plan_option_limit]):
                    if str(e.get('event_id')) == str(remembered_id):
                        default_index = i
                        break
            selected_label = st.selectbox('Choose a concert', options, index=default_index, key='simple_plan_event')
            selected_event = _hydrate_event(plan_pool[options.index(selected_label)])
            st.markdown(f"<div class='agent-context'><b>Selected:</b> {_event_title(selected_event)}<br><span>{_event_venue(selected_event)} · {_event_city(selected_event)}, {_event_state(selected_event)} · {_pretty_when(selected_event)} · {_extract_price(selected_event)}</span></div>", unsafe_allow_html=True)
            n1, n2 = st.columns([1, 2])
            with n1:
                night_style = st.selectbox('Night style', ['auto', 'date night', 'chill', 'high energy', 'casual', 'group night', 'late night'], index=1, key='simple_night_style')
            with n2:
                notes = st.text_area('Anything else?', value='Keep it easy. Good food or drinks nearby, no rushing, and a backup option.', height=90, key='simple_night_notes')
            if st.button('Plan my night', type='primary', key='simple_build_night'):
                with st.spinner('Building the night with venue, city, show time, nearby places, ratings, prices, and your notes...'):
                    _auto_plan(selected_event, night_style, notes, source=source_for_plan)
            _render_plan(st.session_state.get('simple_night_plan'))

# ---------------- Taste ----------------
with main_tabs[3]:
    st.markdown('<div class="section-head">Your Taste</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{user.get("taste_profile_source")}</div>', unsafe_allow_html=True)
    top_visual = [artist for artist in top_artists if artist.get("artist")][:6]
    if top_visual:
        visual_cols = st.columns(len(top_visual))
        for col, artist in zip(visual_cols, top_visual):
            with col:
                with st.container(border=True):
                    if artist.get("image_url"):
                        st.image(artist.get("image_url"), use_container_width=True)
                    st.markdown(f"**{artist.get('artist')}**")
                    st.caption(", ".join((artist.get("genres") or [])[:2]) or "Artist signal")

    artist_col, track_col = st.columns(2)
    with artist_col:
        st.subheader("Top artists")
        columns = [column for column in ["rank", "artist", "genres", "popularity", "blend_score", "time_ranges", "group_listener"] if column in artists_df.columns]
        st.dataframe(artists_df[columns].head(50), use_container_width=True, hide_index=True)
    with track_col:
        st.subheader("Top tracks")
        columns = [column for column in ["rank", "track", "artist", "popularity", "blend_score", "time_ranges"] if column in tracks_df.columns]
        st.dataframe(tracks_df[columns].head(50), use_container_width=True, hide_index=True)

    taste_cluster_profile = build_user_taste_clusters(top_artists, top_tracks)
    cluster_rows = [
        {"cluster": cluster_label(key), "strength": score}
        for key, score in taste_cluster_profile.get("dominant_clusters", [])[:8]
    ]
    if cluster_rows:
        cluster_df = pd.DataFrame(cluster_rows).sort_values("strength", ascending=True)
        st.plotly_chart(
            px.bar(cluster_df, x="strength", y="cluster", orientation="h", title="Your strongest taste clusters"),
            use_container_width=True,
        )
    if st.button("Generate AI taste summary"):
        result = generate_taste_summary(top_artists, top_tracks)
        insert_llm_call("taste_summary", result)
        st.markdown(result.text)


# ---------------- Behind the scenes ----------------
with main_tabs[4]:
    st.markdown('<div class="section-head">Behind the Scenes</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Technical controls and diagnostics for model review, testing, and interviews.</div>', unsafe_allow_html=True)
    lab_tabs = st.tabs(["Coverage", "Model", "Train", "Logs"])

    with lab_tabs[0]:
        raw_total = int(source_counts.get("raw_total", len(ranked_events)))
        deduped_total = int(source_counts.get("deduped_total", len(ranked_events)))
        metrics = st.columns(4)
        metrics[0].metric("Raw candidates", raw_total)
        metrics[1].metric("After dedupe", deduped_total)
        metrics[2].metric("Prices enriched", source_counts.get("price_enriched", 0))
        metrics[3].metric("Price coverage", f"{price_coverage}%")
        by_source = source_counts.get("by_source", {})
        if by_source:
            source_df = pd.DataFrame([{"source": source, "events": count} for source, count in by_source.items()])
            st.plotly_chart(px.bar(source_df, x="source", y="events", title="Candidates by source"), use_container_width=True)
        diagnostic_columns = [column for column in ["event_name", "date", "venue", "source", "min_price", "median_price", "average_price", "price_source", "final_score"] if column in ranked_df.columns]
        st.dataframe(ranked_df[diagnostic_columns].head(250), use_container_width=True, hide_index=True)

    with lab_tabs[1]:
        current_model = load_feedback_model("current")
        previous_model = load_feedback_model("previous")
        if current_model:
            weight = float(current_model.get("recommended_model_weight") or 0.20)
            st.success(
                f"Current learning-to-rank model is automatically active at {weight:.0%} influence. "
                f"It was trained on {current_model.get('n_rows', 0)} ratings across {current_model.get('n_sessions', 0)} sessions."
            )
        else:
            st.info("No promoted feedback model yet. Spotify + genre-cluster ranking is active.")

        model_rows = []
        for name, bundle in [("Current", current_model), ("Previous", previous_model)]:
            if not bundle:
                continue
            metrics = bundle.get("test_metrics", {})
            model_rows.append({
                "model": name,
                "trained_at": bundle.get("trained_at"),
                "rows": bundle.get("n_rows"),
                "sessions": bundle.get("n_sessions"),
                "backend": bundle.get("training_backend"),
                "promoted": bundle.get("promoted"),
                "compared_with": bundle.get("promotion_comparator"),
                "precision@5": metrics.get("model_precision_at_5"),
                "ndcg@5": metrics.get("model_ndcg_at_5"),
                "ndcg@10": metrics.get("model_ndcg_at_10"),
                "not_for_me@10": metrics.get("model_negative_rate_at_10"),
            })
        if model_rows:
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

        st.subheader("Same-event model comparison")
        st.caption("This freezes the concerts already on screen, then shows how the baseline, current model, and previous model would rank the exact same events.")
        comparison = compare_model_variants(ranked_df)
        comparison.insert(0, "event", ranked_df["event_name"].values)
        comparison["live_feed_rank"] = range(1, len(comparison) + 1)
        display_cols = [
            "event", "live_feed_rank", "baseline_rank", "current_rank", "previous_rank",
            "baseline_score", "current_score", "previous_score",
        ]
        display_cols = [column for column in display_cols if column in comparison.columns]
        st.dataframe(comparison[display_cols].head(100), use_container_width=True, hide_index=True)

        preview_cols = st.columns(3)
        variants = [
            ("Baseline top 10", "baseline_rank"),
            ("Current model top 10", "current_rank"),
            ("Previous model top 10", "previous_rank"),
        ]
        for col, (title, rank_col) in zip(preview_cols, variants):
            with col:
                st.markdown(f"**{title}**")
                if rank_col not in comparison.columns or comparison[rank_col].isna().all():
                    st.caption("Not available")
                else:
                    preview = comparison.sort_values(rank_col).head(10)
                    for _, row in preview.iterrows():
                        st.write(f"{int(row[rank_col])}. {row['event']}")

        signal_columns = [column for column in [
            "event_name", "match_confidence", "winning_genre_cluster_label",
            "anchor_artists", "hybrid_score", "model_score", "model_weight", "final_score",
            "genre_cluster_score", "embedding_rank_score", "exact_artist_score",
        ] if column in ranked_df.columns]
        with st.expander("Feature-level ranking signals"):
            st.dataframe(ranked_df[signal_columns].head(100), use_container_width=True, hide_index=True)
    with lab_tabs[2]:
        summary = summarize_feedback()
        active_bundle = load_feedback_model("current")
        current_preferences = int(summary.get("latest_preference_rows", 0))
        trained_rows = int(active_bundle.get("n_rows", 0)) if active_bundle else 0
        new_since_train = max(0, current_preferences - trained_rows)

        train_metrics = st.columns(5)
        train_metrics[0].metric("Current preferences", current_preferences)
        train_metrics[1].metric("Search sessions", summary["unique_sessions"])
        train_metrics[2].metric("Used by current model", trained_rows)
        train_metrics[3].metric("New since training", new_since_train)
        train_metrics[4].metric("Recommended batch", "30–50")

        if current_preferences < 60:
            st.info(f"Collect at least 60 usable ratings before the first training run. You currently have {current_preferences}.")
        elif active_bundle and new_since_train < 30:
            st.info(f"Current model remains active. Add about {30 - new_since_train} more ratings before the next retraining batch.")
        else:
            st.success("You have enough feedback for a training batch. Train one candidate, then review the same-event comparison.")

        with st.expander("Automatic promotion guardrails", expanded=False):
            st.write(
                "A candidate is saved every time, but it is promoted only when the holdout contains at least "
                "15 ratings across 2 search sessions, NDCG@10 and top-5 quality do not decline, Not-for-Me "
                "events do not rise in the top 10, direct Spotify matches are preserved, and the candidate "
                "meaningfully improves on both the baseline and the current promoted model."
            )

        if summary.get("actions"):
            st.dataframe(
                pd.DataFrame([{"action": key, "count": value} for key, value in summary["actions"].items()]),
                use_container_width=True,
                hide_index=True,
            )
        if summary.get("reasons"):
            with st.expander("Feedback reasons"):
                st.dataframe(
                    pd.DataFrame([{"reason": key, "count": value} for key, value in summary["reasons"].items()]),
                    use_container_width=True,
                    hide_index=True,
                )

        controls = st.columns(3)
        with controls[0]:
            if st.button("Train learning-to-rank candidate", type="primary"):
                result = train_feedback_model(min_rows=60)
                if result.get("ok"):
                    st.success(result.get("message"))
                    st.caption(
                        f"Backend: {result.get('training_backend')} · Train rows: {result.get('n_train')} · "
                        f"Holdout rows: {result.get('n_test')} · Holdout sessions: {result.get('n_test_sessions')}"
                    )
                    if result.get("backend_warning"):
                        st.warning(result.get("backend_warning"))
                    if result.get("test_metrics"):
                        metric_rows = [
                            {"metric": key, "value": value}
                            for key, value in result.get("test_metrics", {}).items()
                            if key in {
                                "baseline_precision_at_5", "model_precision_at_5",
                                "baseline_ndcg_at_5", "model_ndcg_at_5",
                                "baseline_ndcg_at_10", "model_ndcg_at_10",
                                "baseline_negative_rate_at_10", "model_negative_rate_at_10",
                            }
                        ]
                        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
                else:
                    st.warning(result.get("message"))
        with controls[1]:
            if st.button("Rollback to previous"):
                result = rollback_to_previous()
                st.success(result["message"]) if result.get("ok") else st.warning(result["message"])
        with controls[2]:
            if st.button("Clear all feedback"):
                clear_all_feedback()
                st.warning("Feedback database cleared.")
                st.rerun()

        versions = list_model_versions()
        if versions:
            st.subheader("Model history")
            st.dataframe(pd.DataFrame(versions), use_container_width=True, hide_index=True)
        feedback_df = get_feedback_df()
        if not feedback_df.empty:
            st.subheader("Recent feedback")
            recent_cols = [column for column in [
                "created_at", "event_name", "action", "feedback_reason",
                "winning_genre_cluster_label", "why_recommended", "session_id",
            ] if column in feedback_df.columns]
            st.dataframe(feedback_df[recent_cols].head(100), use_container_width=True, hide_index=True)
    with lab_tabs[3]:
        calls = load_llm_calls()
        if calls.empty:
            st.info("No Copilot calls logged yet.")
        else:
            st.dataframe(calls, use_container_width=True, hide_index=True)
            if "latency_ms" in calls.columns:
                st.plotly_chart(px.histogram(calls, x="latency_ms", title="Copilot latency"), use_container_width=True)
