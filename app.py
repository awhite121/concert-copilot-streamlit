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


# === PUBLIC PRODUCT MODE V1 ===
def _secret_bool(name: str, default: bool = False) -> bool:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.environ.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


ADMIN_MODE = _secret_bool("ADMIN_MODE", False)
PUBLIC_DEMO_MODE = _secret_bool("PUBLIC_DEMO_MODE", True) and not ADMIN_MODE

# Clear obsolete pagination state only once per browser session.
if not st.session_state.get("_encore_pagination_migrated"):
    for _key in list(st.session_state.keys()):
        if str(_key).endswith("_visible_count"):
            st.session_state.pop(_key, None)
    st.session_state["_encore_pagination_migrated"] = True


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
    for k in ["final_score", "model_score", "score", "recommendation_score", "copilot_score", "rank_score"]:
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


def _cc_score_tier(event):
    score = float(_cc_event_score(event) or 0.0)
    if 0 < score <= 1:
        score *= 100.0
    if score >= 92:
        return "Exceptional match", "tier-exceptional"
    if score >= 84:
        return "Strong match", "tier-strong"
    if score >= 72:
        return "Good discovery", "tier-good"
    if score >= 55:
        return "Possible fit", "tier-possible"
    return "Exploratory", "tier-exploratory"


def _cc_num(value, default=0.0):
    try:
        return float(value if value not in (None, "") else default)
    except Exception:
        return float(default)


def _cc_strength(value):
    value = _cc_num(value)
    if value >= 75:
        return "Excellent"
    if value >= 55:
        return "Strong"
    if value >= 35:
        return "Moderate"
    return "Exploratory"


def _cc_signal_summary(event):
    direct = int(event.get("has_direct_artist_match") or 0) == 1
    artist_value = max(
        _cc_num(event.get("direct_artist_rank_score")),
        _cc_num(event.get("track_affinity_score")),
        _cc_num(event.get("spotify_durability_score")),
    )
    artist_fit = "Excellent" if direct else _cc_strength(artist_value)
    taste_fit = _cc_strength(event.get("genre_cluster_score"))
    discovery_value = max(
        _cc_num(event.get("embedding_rank_score")),
        _cc_num(event.get("discovery_quality_score")),
    )
    return {
        "Artist fit": artist_fit,
        "Taste lane": taste_fit,
        "Discovery": _cc_strength(discovery_value),
    }


def _cc_known_price_text(event):
    value = _cc_price(event)
    return None if value is None else f"From ${value:.0f}"


def _cc_calendar_filename(event):
    title = _cc_event_title(event) or "concert"
    safe = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()
    return f"{safe or 'concert'}.ics"

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
    """Return every currently loaded and filtered show.

    Feedback can change ordering or hidden-state. The previous pagination
    signature treated that as a new result set and reset Discover to 20.
    """
    events = list(events or [])
    if events:
        st.caption(f"Showing all {len(events)} results.")
    return events

def _cc_render_load_more(total, key="discover", page_size=40):
    """Discover displays the complete loaded list, so pagination is disabled."""
    return None

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

st.set_page_config(page_title="Encore AI — Personalized Concerts", page_icon="🎧", layout="wide")
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


@st.cache_data(ttl=2700, show_spinner=False)
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
.mini-score{font-size:.76rem;color:#64748b;margin-top:.55rem;font-weight:700}.mini-card{border:1px solid rgba(15,23,42,.10);border-radius:14px;padding:14px;background:#fff;min-height:104px;box-shadow:0 8px 24px rgba(15,23,42,.04)}.mini-card span{color:#64748b;font-size:.85rem}
/* Public product makeover */
.product-profile {display:flex;align-items:center;justify-content:space-between;gap:18px;background:#fff;border:1px solid #e6e8ef;border-radius:22px;padding:14px 18px;margin:0 0 16px;box-shadow:0 12px 28px rgba(17,24,39,.05)}
.profile-left {display:flex;align-items:center;gap:12px;min-width:0}.profile-avatar{width:48px;height:48px;border-radius:50%;object-fit:cover;border:2px solid #fff;box-shadow:0 0 0 2px #e7eaf1}.profile-fallback{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#ff6b63,#ff9b78);color:white;font-weight:900;font-size:18px}.profile-name{font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.08rem;font-weight:850;color:var(--ink)}.profile-meta{font-size:.82rem;color:#727a8c;margin-top:2px}.profile-connected{display:inline-flex;align-items:center;gap:7px;background:#effaf4;border:1px solid #c5ead2;color:#20723f;border-radius:999px;padding:7px 11px;font-size:.78rem;font-weight:850;white-space:nowrap}
.hero-feature-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:18px;position:relative;z-index:1}.hero-feature{background:#f8f9fc;border:1px solid #e9ebf2;border-radius:15px;padding:12px 14px}.hero-feature strong{display:block;color:#252b38;font-size:.88rem;margin-bottom:2px}.hero-feature span{font-size:.77rem;color:#7a8295}
.top-picks-title{font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.4rem;font-weight:850;color:var(--ink);letter-spacing:-.03em;margin:.35rem 0 .08rem}.top-picks-sub{font-size:.9rem;color:#7a8295;margin-bottom:.65rem}.top-pick-card{background:linear-gradient(155deg,#fff 0%,#faf8ff 100%);border:1px solid #e7e5ef;border-radius:20px;padding:15px;height:100%;min-height:188px;box-shadow:0 12px 28px rgba(17,24,39,.05)}.top-pick-label{font-size:.69rem;letter-spacing:.11em;text-transform:uppercase;font-weight:900;color:#ff5954}.top-pick-title{font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.08rem;line-height:1.1;font-weight:850;color:#1d2230;margin:8px 0 6px}.top-pick-meta{font-size:.78rem;color:#737b8c;line-height:1.4}.top-pick-score{display:inline-flex;margin-top:10px;border-radius:999px;background:#edf9f1;border:1px solid #c4e8ce;color:#24643a;padding:5px 8px;font-size:.73rem;font-weight:850}
.quick-filter-title{font-size:.76rem;text-transform:uppercase;letter-spacing:.1em;color:#9097a5;font-weight:900;margin-top:.9rem}.share-copy{font-size:.8rem;color:#777f90}.timeline-card{background:linear-gradient(160deg,#fff,#f8f9fc);border:1px solid #e5e8ef;border-radius:18px;padding:15px;min-height:132px;box-shadow:0 10px 25px rgba(17,24,39,.045)}.timeline-time{font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:#ff5954;font-weight:900}.timeline-step{font-size:1rem;font-weight:850;color:#202635;margin:7px 0 4px}.timeline-detail{font-size:.84rem;line-height:1.42;color:#697184}
[data-testid="stTabs"] button[role="tab"]{font-weight:850!important;font-size:.92rem!important;padding-left:18px!important;padding-right:18px!important}
@media(max-width:800px){.hero-feature-grid{grid-template-columns:1fr}.product-profile{align-items:flex-start}.profile-connected{display:none}.top-pick-card{min-height:160px}}


/* Public UI refinement */
.top-pick-card{
  overflow:hidden;
  padding:0;
  min-height:292px;
  display:flex;
  flex-direction:column;
}
.top-pick-image{
  width:100%;
  height:108px;
  object-fit:cover;
  display:block;
  background:linear-gradient(135deg,#f1ebe4,#f7f4ef);
}
.top-pick-image-fallback{
  width:100%;
  height:108px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:linear-gradient(135deg,#fff0ee,#f5f2ff);
  color:#ff5954;
  font-size:.72rem;
  font-weight:900;
  letter-spacing:.13em;
  text-transform:uppercase;
}
.top-pick-body{padding:14px 15px 15px;display:flex;flex-direction:column;flex:1}
.top-pick-reason{
  margin-top:9px;
  font-size:.78rem;
  line-height:1.38;
  color:#687083;
  display:-webkit-box;
  -webkit-line-clamp:2;
  -webkit-box-orient:vertical;
  overflow:hidden;
}
.recommendation-heading{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:16px;
  margin:1.15rem 0 .25rem;
}
.recommendation-heading-title{
  font-family:'Bricolage Grotesque',Inter,sans-serif;
  font-size:1.65rem;
  font-weight:850;
  letter-spacing:-.035em;
  color:var(--ink);
}
.recommendation-heading-copy{font-size:.9rem;color:#737b8c;margin-top:3px}
.recommendation-count{
  display:inline-flex;
  border-radius:999px;
  background:#f5f6fa;
  border:1px solid #e7eaf1;
  color:#626a7b;
  padding:7px 11px;
  font-size:.78rem;
  font-weight:850;
  white-space:nowrap;
}
.save-label{
  margin:.75rem 0 .38rem;
  color:#8a92a3;
  font-size:.71rem;
  font-weight:900;
  text-transform:uppercase;
  letter-spacing:.11em;
}
.card-action-spacer{height:.2rem}
div[data-testid="stVerticalBlockBorderWrapper"]{
  transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{
  transform:translateY(-2px);
  box-shadow:0 22px 48px rgba(17,24,39,.095)!important;
  border-color:#dce0e9!important;
}
@media(max-width:900px){
  .recommendation-heading{align-items:flex-start;flex-direction:column}
  .top-pick-card{min-height:250px}
}


.filter-panel-title{
  margin:1rem 0 .45rem;
  color:#252b38;
  font-size:.82rem;
  font-weight:900;
  letter-spacing:.08em;
  text-transform:uppercase;
}

/* Encore AI Elite V1 */
.match-tier{display:inline-flex;border-radius:999px;padding:6px 10px;font-size:.75rem;font-weight:850;border:1px solid transparent}
.tier-exceptional{background:#171b26;color:#fff;border-color:#171b26}
.tier-strong{background:#edf9f1;color:#24643a;border-color:#c4e8ce}
.tier-good{background:#eef3ff;color:#335fa8;border-color:#d4def5}
.tier-possible{background:#fff6e9;color:#9a5d12;border-color:#f0d8b3}
.tier-exploratory{background:#f4f5f8;color:#687083;border-color:#e2e5ec}
.fit-signal-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:.75rem 0 .15rem}
.fit-signal{background:#f8f9fc;border:1px solid #e7eaf1;border-radius:14px;padding:9px 11px}
.fit-signal-label{font-size:.67rem;text-transform:uppercase;letter-spacing:.09em;color:#8a92a3;font-weight:900}
.fit-signal-value{font-size:.84rem;color:#252b38;font-weight:850;margin-top:2px}
.elite-decision{background:linear-gradient(135deg,#171b26,#2a3040);color:#fff;border-radius:22px;padding:18px 20px;margin:.8rem 0 1rem;box-shadow:0 18px 38px rgba(17,24,39,.16)}
.elite-decision-label{font-size:.69rem;letter-spacing:.13em;text-transform:uppercase;color:#ff8d87;font-weight:900}
.elite-decision-title{font-family:'Bricolage Grotesque',Inter,sans-serif;font-size:1.35rem;font-weight:850;letter-spacing:-.025em;margin:6px 0 4px}
.elite-decision-copy{font-size:.9rem;line-height:1.45;color:#d8dce5}
.copilot-tradeoff{font-size:.78rem;line-height:1.4;color:#7a8295;margin-top:8px}
.plan-control-shell{background:#f8f9fc;border:1px solid #e7eaf1;border-radius:18px;padding:14px 16px;margin:.65rem 0 1rem}
@media(max-width:800px){.fit-signal-grid{grid-template-columns:1fr}.match-tier{margin-top:4px}}

</style>
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
    st.markdown('''
    <div class="new-hero">
      <div class="hero-eyebrow">Concert discovery, personalized</div>
      <div class="hero-h1">Find concerts that actually match <span style="color:#ff5954;">your taste.</span></div>
      <div class="hero-sub">Connect Spotify, choose a city, and get a short list of shows worth caring about — with clear reasons, a personal playlist, and help planning the night.</div>
      <div class="hero-feature-grid">
        <div class="hero-feature"><strong>Personalized by Spotify</strong><span>Your artists, tracks, and listening patterns.</span></div>
        <div class="hero-feature"><strong>Ranked by music fit</strong><span>Direct matches plus relevant discoveries.</span></div>
        <div class="hero-feature"><strong>Save and plan the night</strong><span>Shortlist shows, compare picks, and build a plan.</span></div>
      </div>
    </div>
    ''', unsafe_allow_html=True)


def render_profile_header(user: Dict[str, Any], top_artists: List[Dict[str, Any]]) -> None:
    name = str(user.get("display_name") or "Spotify listener")
    image = user.get("image_url")
    if not image:
        image = next((a.get("image_url") for a in (top_artists or []) if a.get("image_url")), None)
    initial = escape(name[:1].upper() if name else "E")
    avatar = (
        f'<img class="profile-avatar" src="{escape(str(image))}" alt="{escape(name)}">'
        if image else f'<div class="profile-fallback">{initial}</div>'
    )
    source = "Spotify connected" if user.get("user_id") != "demo_user" else "Demo taste profile"
    st.markdown(
        f'''<div class="product-profile">
          <div class="profile-left">{avatar}<div><div class="profile-name">{escape(name)}</div><div class="profile-meta">Taste refreshed today · Personalized recommendations ready</div></div></div>
          <div class="profile-connected"><span style="width:7px;height:7px;border-radius:50%;background:#1DB954;display:inline-block"></span>{escape(source)}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def _share_event_text(event: Dict[str, Any]) -> str:
    return (
        "My Encore AI concert pick:\n"
        f"{_cc_event_title(event)} — {_cc_event_venue(event)} — {format_when(event)}\n"
        f"{_cc_match_score_label(event)}"
    )


def _is_weekend_event(event: Dict[str, Any]) -> bool:
    if int(event.get("weekend_event") or 0) == 1:
        return True
    try:
        return datetime.strptime(str(event.get("date"))[:10], "%Y-%m-%d").weekday() >= 4
    except Exception:
        return False


def apply_quick_filter(events: List[Dict[str, Any]], quick_filter: str) -> List[Dict[str, Any]]:
    rows = list(events or [])
    choice = str(quick_filter or "Best for me")
    if choice == "This weekend":
        rows = [e for e in rows if _is_weekend_event(e)]
    elif choice == "Direct matches":
        rows = [e for e in rows if int(e.get("has_direct_artist_match") or 0) == 1]
    elif choice == "New discoveries":
        rows = [e for e in rows if int(e.get("has_direct_artist_match") or 0) == 0]
    elif choice == "Date night":
        rows = [e for e in rows if _is_weekend_event(e) or float(e.get("venue_quality_signal") or 0) >= 45]
    elif choice == "Under $50":
        rows = [e for e in rows if isinstance(e.get("min_price"), (int, float)) and float(e.get("min_price")) <= 50]
    elif choice == "Intimate venues":
        words = ("club", "room", "parish", "mohawk", "scoot", "antone", "lounge", "bar", "theatre", "theater", "hall")
        rows = [e for e in rows if any(w in _cc_event_venue(e).lower() for w in words)]
    return sorted(rows, key=lambda e: _cc_event_score(e), reverse=True)


def _unique_pick(candidates: List[Dict[str, Any]], used: set) -> Dict[str, Any] | None:
    for event in candidates:
        key = _cc_event_id(event) or (_cc_event_title(event), str(event.get("date")))
        if key not in used:
            used.add(key)
            return event
    return None




def render_top_picks(events: List[Dict[str, Any]], session_id: str) -> None:
    events = sorted(_dedupe_events_for_display(events or []), key=lambda e: _cc_event_score(e), reverse=True)
    if not events:
        return

    used = set()
    direct = sorted(
        [e for e in events if int(e.get("has_direct_artist_match") or 0) == 1],
        key=lambda e: _cc_event_score(e),
        reverse=True,
    )
    discovery = sorted(
        [e for e in events if int(e.get("has_direct_artist_match") or 0) == 0],
        key=lambda e: _cc_num(e.get("discovery_quality_score")) * .45 + _cc_num(e.get("genre_cluster_score")) * .25 + _cc_event_score(e) * .30,
        reverse=True,
    )
    night_out = sorted(
        events,
        key=lambda e: _cc_event_score(e) * .58 + _cc_num(e.get("venue_quality_signal")) * .24 + (12 if _is_weekend_event(e) else 0) + (6 if _cc_price(e) is not None else 0),
        reverse=True,
    )
    priced = [e for e in events if _cc_price(e) is not None]
    value = sorted(
        priced,
        key=lambda e: _cc_event_score(e) * .70 + _cc_num(e.get("price_score")) * .25 + min(_cc_num(e.get("source_count"), 1), 3) * 2,
        reverse=True,
    )
    weekend = sorted([e for e in events if _is_weekend_event(e)], key=lambda e: _cc_event_score(e), reverse=True)

    definitions = [
        ("Best match", events),
        ("Artist you know", direct or events),
        ("Discovery pick", discovery or events),
        ("Best value" if value else "Weekend pick", value or weekend or events),
        ("Best night out", night_out),
    ]

    picks = []
    for label, pool in definitions:
        pick = _unique_pick(pool, used)
        if pick:
            picks.append((label, pick))

    st.markdown(
        '<div class="top-picks-title">The five shows to know</div>'
        '<div class="top-picks-sub">A short list built for different ways you might want to spend the night.</div>',
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(picks), 3):
        row_picks = picks[row_start:row_start + 3]
        cols = st.columns(len(row_picks))
        for local_idx, ((label, event), col) in enumerate(zip(row_picks, cols)):
            idx = row_start + local_idx
            with col:
                title = escape(_cc_event_title(event) or "Concert")
                meta = escape(f"{format_when(event)} · {_cc_event_venue(event)}")
                image_url = event.get("image_url")
                image_html = f'<img class="top-pick-image" src="{escape(str(image_url))}" alt="{title}">' if image_url else '<div class="top-pick-image-fallback">Encore pick</div>'
                reason = str(event.get("why_artist_match") or event.get("why_recommended") or f"A strong match for your {_cc_display_lane(event)} taste.").strip()
                if len(reason) > 150:
                    reason = reason[:147].rstrip() + "…"
                tier, tier_class = _cc_score_tier(event)
                price_text = _cc_known_price_text(event)
                price_html = f'<div class="top-pick-meta">{escape(price_text)}</div>' if price_text else ""

                st.markdown(
                    f'''<div class="top-pick-card">{image_html}<div class="top-pick-body">
                    <div class="top-pick-label">{escape(label)}</div><div class="top-pick-title">{title}</div>
                    <div class="top-pick-meta">{meta}</div>{price_html}
                    <div style="display:flex;gap:7px;flex-wrap:wrap;margin-top:10px">
                    <div class="top-pick-score">{escape(_cc_match_score_label(event))}</div>
                    <div class="match-tier {tier_class}">{escape(tier)}</div></div>
                    <div class="top-pick-reason">{escape(reason)}</div></div></div>''',
                    unsafe_allow_html=True,
                )
                action_a, action_b = st.columns(2)
                links = event_links(event)
                with action_a:
                    if links:
                        st.link_button("Tickets", links[0], use_container_width=True)
                with action_b:
                    if st.button("Plan", key=f"top_pick_plan_{session_id}_{idx}", use_container_width=True):
                        st.session_state.plan_event_id = event.get("event_id")
                        st.session_state.plan_source_request = "Recommended list"
                        st.toast("Ready in Copilot → Plan a Night")

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
    title = event.get("event_name") or "Untitled event"
    venue_line = f"{event.get('venue') or 'Venue TBD'} · {event.get('city') or ''}, {event.get('state') or ''}".strip(" ·,")
    when = format_when(event)
    lane = _cc_display_lane(event)
    confidence = event.get("match_confidence") or "Taste match"
    why = event.get("why_artist_match") or event.get("why_recommended") or "This event matches your broader listening profile."
    lane_copy = event.get("why_taste_lane") or ""
    reason_key = f"reason_{section}_{session_id}_{event.get('event_id')}_{idx}"
    links = event_links(event)
    tier, tier_class = _cc_score_tier(event)
    signals = _cc_signal_summary(event)

    spotify_links = event.get("artist_spotify_urls") or []
    spotify_url = event.get("spotify_url") or _cc_spotify_url(event)
    spotify_label = event.get("spotify_label") or _cc_spotify_label(event)
    if spotify_links and spotify_links[0].get("url"):
        spotify_url = spotify_links[0].get("url")
        spotify_label = f"Spotify · {spotify_links[0].get('artist') or 'Artist'}"

    with st.container(border=True):
        poster_col, content_col = st.columns([0.18, 0.82], vertical_alignment="top")
        with poster_col:
            if event.get("image_url"):
                st.markdown(f'<div class="poster-wrap"><img src="{escape(str(event.get("image_url")))}" alt="event poster"></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="poster-fallback">Event Poster</div>', unsafe_allow_html=True)

        with content_col:
            st.markdown(
                '<div class="card-topline">'
                f'<span class="rank-chip">#{idx}</span><span class="card-date">{escape(when)}</span></div>'
                f'<div class="card-title">{escape(str(title))}</div><div class="card-venue">{escape(venue_line)}</div>',
                unsafe_allow_html=True,
            )

            signal_parts = [
                f'<span class="signal-pill badge-price">{escape(_cc_match_score_label(event))}</span>',
                f'<span class="match-tier {tier_class}">{escape(tier)}</span>',
                f'<span class="signal-pill"><span class="signal-dot dot-coral"></span>{escape(str(confidence))}</span>',
            ]
            if extra_badge:
                signal_parts.append(f'<span class="signal-pill"><span class="signal-dot dot-blue"></span>{escape(str(extra_badge))}</span>')
            signal_parts.append(f'<span class="signal-pill no-dot">{escape(str(lane))}</span>')
            signal_cards = "".join(
                f'<div class="fit-signal"><div class="fit-signal-label">{escape(label)}</div><div class="fit-signal-value">{escape(value)}</div></div>'
                for label, value in signals.items()
            )
            recommendation_copy = str((why + " " + lane_copy).strip())
            st.markdown(
                '<div class="signal-row">' + "".join(signal_parts) + '</div>'
                '<div class="fit-signal-grid">' + signal_cards + '</div>'
                '<div class="why-note"><div class="why-label">Why it fits</div>'
                f'<div class="why-copy">{escape(recommendation_copy)}</div></div>',
                unsafe_allow_html=True,
            )

            if ADMIN_MODE:
                with st.popover("Feedback reasons"):
                    st.multiselect("Pick any that apply", FEEDBACK_REASON_OPTIONS, key=reason_key)

            primary_actions = st.columns([1.05, 1.0, 1.0, 1.2])
            with primary_actions[0]:
                if links:
                    st.link_button("Tickets →", links[0], type="primary", use_container_width=True)
            with primary_actions[1]:
                if st.button("Plan night", key=f"plan_{section}_{session_id}_{event.get('event_id')}_{idx}", use_container_width=True):
                    st.session_state.plan_event_id = event.get("event_id")
                    st.session_state.plan_source_request = "Recommended list"
                    st.toast("Ready in Copilot → Plan a Night")
            with primary_actions[2]:
                st.download_button(
                    "Add to calendar",
                    data=build_calendar_ics(event),
                    file_name=_cc_calendar_filename(event),
                    mime="text/calendar",
                    key=f"calendar_{section}_{session_id}_{event.get('event_id')}_{idx}",
                    use_container_width=True,
                )
            with primary_actions[3]:
                st.link_button(spotify_label, spotify_url, use_container_width=True)

            st.markdown('<div class="save-label">Save to your playlist</div>', unsafe_allow_html=True)
            save_actions = st.columns(3)
            with save_actions[0]:
                if st.button("Want to go", key=f"want_{section}_{session_id}_{event.get('event_id')}_{idx}", use_container_width=True):
                    save_feedback_action(user, session_id, event, "want_to_go", rank_position=idx, feedback_reasons=_feedback_reason_value(reason_key))
                    st.toast("Saved to Want")

            with save_actions[1]:
                if st.button("Maybe", key=f"maybe_{section}_{session_id}_{event.get('event_id')}_{idx}", use_container_width=True):
                    save_feedback_action(user, session_id, event, "maybe", rank_position=idx, feedback_reasons=_feedback_reason_value(reason_key))
                    st.toast("Saved to Maybe")

            with save_actions[2]:
                if st.button("Not for me", key=f"no_{section}_{session_id}_{event.get('event_id')}_{idx}", use_container_width=True):
                    save_feedback_action(user, session_id, event, "not_for_me", rank_position=idx, feedback_reasons=_feedback_reason_value(reason_key))
                    st.toast("Hidden from recommendations")
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

    # Always use complete coverage.
    # Speed comes from caching, parallel requests, and rendering only 20 cards first.
    search_speed = "Full coverage"
    speed_config = {
        "size": 250,
        "ticketmaster_pages": 5,
        "seatgeek_pages": 5,
        "top_artist_search_count": 40,
        "price_enrichment_limit": 16,
        "display_page_size": 20,
    }
    if ADMIN_MODE:
        st.caption("Full coverage is always enabled.")

    group_mode = False
    group_name = "Friend"
    group_artist_text = ""
    with st.expander("Going with someone?"):
        group_mode = st.checkbox("Blend our music taste", value=False)
        if group_mode:
            group_name = st.text_input("Their name", "Friend")
            group_artist_text = st.text_area("A few artists they love", "", height=70, placeholder="Mt. Joy, Noah Kahan, Fred again..")
            st.caption("Encore AI will look for shows that work for both of you.")

    refresh_taste = False
    top_artist_search_count = int(speed_config["top_artist_search_count"])
    use_feedback_model_toggle = True
    use_saved_history_toggle = True

    if ADMIN_MODE:
        with st.expander("Admin controls"):
            refresh_taste = st.checkbox("Refresh Spotify taste", value=False)
            top_artist_search_count = st.slider("Top Spotify artists searched directly", 10, 40, top_artist_search_count, step=5)
            use_feedback_model_toggle = st.checkbox("Use feedback model", value=True)
            use_saved_history_toggle = st.checkbox("Use saved history", value=True)

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

    if ADMIN_MODE:
        status = model_status()
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

render_profile_header(user, top_artists)

active_model = run_data.get("filters", {}).get("model_active")
bundle = load_feedback_model("current") if active_model else None
# V29 keeps noisy dashboard stats out of the UI. Keep only the lightweight model badge/status.
ranked_events = _dedupe_events_for_display(ranked_events)
if ADMIN_MODE:
    render_stats_strip(len(ranked_events), direct_count, price_coverage, bool(active_model), bundle)

# Playlist memory is shared across Discover, My Playlist, and Copilot unless test mode disables it.
if use_saved_history_toggle:
    playlist_memory_df = load_playlist_preferences(user.get("user_id", "unknown_user"))
else:
    playlist_memory_df = pd.DataFrame()
playlist_memory_events = [] if playlist_memory_df.empty else [interaction_row_to_event(row) for row in playlist_memory_df.to_dict(orient="records")]
active_playlist_events = [] if playlist_memory_df.empty else [interaction_row_to_event(row) for row in playlist_memory_df[playlist_memory_df["action"].isin(["want_to_go", "maybe"])].to_dict(orient="records")]

if ADMIN_MODE and source_counts.get("errors"):
    with st.expander("Source warnings"):
        st.json(source_counts.get("errors"))

tab_labels = ["Discover", "Playlist", "Copilot", "Taste"]
if ADMIN_MODE:
    tab_labels.append("Behind the Scenes")
main_tabs = st.tabs(tab_labels)




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
    render_top_picks(base_events, session_id)
    venues = sorted({ _cc_event_venue(e) for e in base_events if _cc_event_venue(e) })
    city_values_current = sorted({ _cc_event_city(e) for e in base_events if _cc_event_city(e) })
    genre_options = ["All genres"] + sorted({str(_cc_genre(e)) for e in base_events if _cc_genre(e)})

    st.markdown(
        f'''<div class="recommendation-heading">
              <div>
                <div class="recommendation-heading-title">All recommendations</div>
                <div class="recommendation-heading-copy">Browse the full ranked list, narrow it down, or save a show for later.</div>
              </div>
              <div class="recommendation-count">{len(base_events)} shows ranked</div>
            </div>''',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="quick-filter-title">Browse by fit</div>', unsafe_allow_html=True)
    quick_filter = st.radio(
        "Explore",
        ["Best for me", "This weekend", "Direct matches", "New discoveries", "Date night", "Under $50", "Intimate venues"],
        horizontal=True,
        label_visibility="collapsed",
        key="discover_quick_filter_public",
    )

    discover_search_text = ""
    discover_city = "All cities"
    display_venue = "All venues"
    discover_genre = "All genres"
    show_hidden = False
    st.markdown('<div class="filter-panel-title">Search & filters</div>', unsafe_allow_html=True)
    with st.container(border=True):
        f_search, f_city, f_venue, f_genre, f_hidden = st.columns([2.35, .95, 1.15, 1.05, .65], vertical_alignment="bottom")
        with f_search:
            discover_search_text = st.text_input(
                "Search artist, show, genre, venue",
                placeholder="Type an artist, venue, or genre",
                key="discover_search_text_v40",
            )
        with f_city:
            discover_city = st.selectbox("City", ["All cities"] + city_values_current, key="discover_city_filter_v40")
        with f_venue:
            display_venue = st.selectbox("Venue", ["All venues"] + venues, key="discover_venue_filter_v40")
        with f_genre:
            discover_genre = st.selectbox("Genre", genre_options, key="discover_genre_v40")
        with f_hidden:
            show_hidden = st.checkbox(
                "Hidden",
                value=False,
                help="Show events marked Not for me.",
                key="discover_hidden_v40",
            )

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

    visible = apply_quick_filter(visible, quick_filter)

    if use_saved_history_toggle:
        # Normal product mode: saved Want first, saved Maybe second, then Match Score.
        visible = _cc_sort_saved_first(visible, status_by_id)
        caption = f"Showing {len(visible)} of {len(base_events)} loaded shows. Saved Want and Maybe stay first, then new shows rank by Match Score."
    else:
        # Clean test mode: ignore saved-first memory and rank by current recommender score only.
        visible = sorted(visible, key=lambda e: _cc_event_score(e), reverse=True)
        caption = f"Clean test: showing {len(visible)} of {len(base_events)} loaded shows. Saved history is ignored; current ranking score only."

    visible = _dedupe_events_for_display(visible)
    visible_total = len(visible)
    visible = _cc_limit_visible_events(visible, key=f"discover_{session_id}", page_size=display_page_size)

    if not visible:
        st.info("No concerts match the current filters. Clear search, widen filters, or turn on Hidden.")
    for index, event in enumerate(visible, 1):
        saved_badge = preference_label(status_by_id.get(str(event.get("event_id")))) if use_saved_history_toggle else None
        extra = saved_badge or event.get("group_fit_label")
        render_event_card(event, index, "feed", user, session_id, extra)

    _cc_render_load_more(
        visible_total,
        key=f"discover_{session_id}",
        page_size=display_page_size,
    )


# ---------------- Playlist ----------------
with main_tabs[1]:
    st.markdown('<div class="section-head">My Playlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Your personal shortlist of concerts you want to remember, compare, or plan.</div>', unsafe_allow_html=True)
    playlist = load_playlist_preferences(user.get("user_id", "unknown_user")) if use_saved_history_toggle else pd.DataFrame()

    if not use_saved_history_toggle:
        st.info("Your playlist is unavailable in this session.")
    elif playlist.empty:
        st.info("Your playlist is empty. Save a show as Want or Maybe from Discover to add it here.")
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
                        <span class="badge">{escape(str(_cc_display_lane(event)))}</span>
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
    st.markdown('<div class="section-sub">Tell Encore what kind of night you want. It will compare your ranked shows and give you a clear recommendation.</div>', unsafe_allow_html=True)

    if ADMIN_MODE:
        copilot_speed_mode = st.radio(
            "Copilot response",
            ["Quick answer", "Detailed answer"],
            horizontal=True,
            index=0,
            help="Detailed answer adds a longer grounded explanation.",
            key="copilot_speed_mode_v1",
        )
    else:
        copilot_speed_mode = "Quick answer"
    copilot_fast_mode = copilot_speed_mode == "Quick answer"

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
            'Best choice': 'request_match',
            'Artist you know': 'top_artist',
            'Best discovery': 'discovery',
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


    def _plain_event_card(event, label, score=None, context=None, button_key_suffix=''):
        event = _hydrate_event(event or {})
        context = context or {}
        title = _event_title(event)
        venue = _event_venue(event)
        city = _event_city(event)
        state = _event_state(event)
        when = _pretty_when(event)
        why = str(event.get('why_recommended') or event.get('why_artist_match') or 'Strong fit based on your taste and event context.')
        tier, tier_class = _cc_score_tier(event)
        tradeoffs = context.get('tradeoffs') or []
        tradeoff_text = "; ".join(str(x) for x in tradeoffs[:2])
        price_text = _cc_known_price_text(event)
        score_html = f"<span class='badge badge-price'>Decision score {float(score):.1f}</span>" if isinstance(score, (int, float)) else ""
        price_html = f"<span class='badge'>{escape(price_text)}</span>" if price_text else ""
        trade_html = f"<div class='copilot-tradeoff'><b>Tradeoff:</b> {escape(tradeoff_text)}.</div>" if tradeoff_text else "<div class='copilot-tradeoff'><b>Tradeoff:</b> No major caveat in the returned data.</div>"
        st.markdown(
            f'''<div class="shortlist-card"><div class="badge badge-direct">{escape(str(label))}</div>
            <div class="shortlist-title">{escape(title)}</div><div class="shortlist-meta"><b>{escape(when)}</b> · {escape(venue)} · {escape(city)}, {escape(state)}</div>
            <span class="match-tier {tier_class}">{escape(tier)}</span>{score_html}{price_html}
            <div class="shortlist-reason"><b>Why it works:</b> {escape(why)}</div>{trade_html}</div>''',
            unsafe_allow_html=True,
        )
        url = _ticket_url(event)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button('Plan this night', key=f"auto_plan_{button_key_suffix}_{label}_{event.get('event_id') or title}", use_container_width=True):
                with st.spinner('Building your plan...'):
                    _auto_plan(event, source='Copilot picks')
        with b2:
            if url:
                st.link_button('Tickets', url, use_container_width=True)
        with b3:
            st.download_button('Add to calendar', data=build_calendar_ics(event), file_name=_cc_calendar_filename(event), mime='text/calendar', key=f"copilot_calendar_{button_key_suffix}_{event.get('event_id') or title}", use_container_width=True)

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
                        st.markdown(f"<div class='timeline-card'><div class='timeline-time'>{t}</div><div class='timeline-step'>{title}</div><div class='timeline-detail'>{place}</div></div>", unsafe_allow_html=True)
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

        goal_defaults = {
            'Best show for me': 'Recommend the single best show for me and give me two strong alternatives.',
            'Plan a date night': 'Find the best concert for a date night and explain what makes the full night work.',
            'This weekend': 'What should I see this weekend? Prioritize timing, music fit, and an easy night out.',
            'Smaller venue': 'Find me a great show at a smaller or more intimate venue.',
            'Compare top 3': 'Compare my top three concert options and tell me which one you would choose.',
        }
        if 'simple_copilot_prompt_value' not in st.session_state:
            st.session_state.simple_copilot_prompt_value = goal_defaults['Best show for me']
        st.markdown('<div class="quick-filter-title">Try asking</div>', unsafe_allow_html=True)
        prompt_cols = st.columns(5)
        for prompt_idx, (label, prompt_text) in enumerate(goal_defaults.items()):
            with prompt_cols[prompt_idx]:
                if st.button(label, key=f"copilot_prompt_{prompt_idx}", use_container_width=True):
                    st.session_state.simple_copilot_prompt_value = prompt_text
        user_question = st.text_area(
            'What are you looking for?',
            height=105,
            key='simple_copilot_prompt_value',
            placeholder='Ask for a date night, a weekend show, a discovery, or a comparison...',
        )

        if source_choice == 'Current search':
            base_events = _current_events
        elif source_choice == 'My Playlist':
            base_events = _saved_events or _playlist_events
        else:
            base_events = _all_memory_events
        city_events = _filter_city(base_events, selected_city)
        st.caption(f"Using {len(city_events)} shows after city/source filtering.")

        if st.button('Find my best shows', type='primary', use_container_width=True, key='find_simple_copilot'):
            if not city_events:
                st.warning('No shows match that city/source yet. Run a search for that city or switch source to Both/Current search.')
            else:
                with st.spinner('Finding your best options...' if copilot_fast_mode else 'Building your detailed answer...'):
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
                        labels = ['Best choice', 'Artist you know', 'Best discovery', 'Best night out']
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
            priority = ["Best choice", "Artist you know", "Best discovery", "Best night out"]
            display_picks = [(label, picks.get(label)) for label in priority if picks.get(label)]
            if not display_picks:
                display_picks = [(label, ctx) for label, ctx in picks.items() if ctx][:4]

            if display_picks:
                _, best_ctx = display_picks[0]
                best_event = _hydrate_event(best_ctx.get('event', best_ctx) if isinstance(best_ctx, dict) else best_ctx)
                best_reason = str(best_event.get('why_recommended') or best_event.get('why_artist_match') or 'It has the strongest combination of taste fit and event context.')
                st.markdown(
                    f'''<div class="elite-decision"><div class="elite-decision-label">Encore's decision</div>
                    <div class="elite-decision-title">{escape(_event_title(best_event))}</div>
                    <div class="elite-decision-copy">{escape(best_reason)}</div></div>''',
                    unsafe_allow_html=True,
                )

            cols = st.columns(min(4, max(1, len(display_picks))))
            for idx, (label, ctx) in enumerate(display_picks):
                event = _hydrate_event(ctx.get('event', ctx) if isinstance(ctx, dict) else ctx)
                score = _safe_score(ctx, label) if isinstance(ctx, dict) else (event or {}).get('final_score')
                with cols[idx % len(cols)]:
                    _plain_event_card(event, label, score, context=ctx if isinstance(ctx, dict) else None, button_key_suffix='ask')
            if st.session_state.get('simple_night_plan'):
                st.markdown('---')
                _render_plan(st.session_state.get('simple_night_plan'))
            narrative = result.get('narrative')
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
            action_cols = st.columns(3)
            ticket_url = _ticket_url(selected_event)
            with action_cols[0]:
                if ticket_url:
                    st.link_button('Tickets', ticket_url, use_container_width=True)
            with action_cols[1]:
                st.download_button('Add to calendar', data=build_calendar_ics(selected_event), file_name=_cc_calendar_filename(selected_event), mime='text/calendar', key='selected_plan_calendar', use_container_width=True)
            with action_cols[2]:
                st.link_button('Spotify', selected_event.get('spotify_url') or _cc_spotify_url(selected_event), use_container_width=True)

            n1, n2, n3 = st.columns(3)
            with n1:
                vibe_label = st.selectbox('Vibe', ['Date night', 'Friends', 'Low-key', 'High-energy', 'Budget-friendly', 'Upscale'], index=0, key='simple_night_style')
            with n2:
                budget_label = st.selectbox('Budget', ['Flexible', 'Keep it affordable', 'Under $75 before tickets', 'Special occasion'], index=0, key='simple_night_budget')
            with n3:
                travel_label = st.selectbox('Getting around', ['Keep everything close', 'Walking preferred', 'Rideshare is fine', 'Driving'], index=0, key='simple_night_travel')
            notes = st.text_area('Anything else?', value='Good food or drinks nearby, no rushing, and one flexible backup option.', height=85, key='simple_night_notes')
            style_map = {'Date night': 'date night', 'Friends': 'group night', 'Low-key': 'chill', 'High-energy': 'high energy', 'Budget-friendly': 'casual', 'Upscale': 'date night'}
            night_style = style_map.get(vibe_label, 'date night')
            combined_notes = f"{notes} Budget preference: {budget_label}. Transportation preference: {travel_label}."
            if st.button('Build my night', type='primary', key='simple_build_night', use_container_width=True):
                with st.spinner('Building your night...'):
                    _auto_plan(selected_event, night_style, combined_notes, source=source_for_plan)
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
if ADMIN_MODE:
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
