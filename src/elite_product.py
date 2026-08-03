from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math
import re
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st


INTIMATE_VENUE_WORDS = (
    "club", "room", "parish", "mohawk", "scoot", "antone",
    "lounge", "bar", "theatre", "theater", "hall", "ballroom",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        value = float(value)
        if math.isnan(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def _norm(value: Any) -> str:
    return "".join(ch.lower() for ch in _text(value) if ch.isalnum())


def event_title(event: Dict[str, Any]) -> str:
    for key in ("event_name", "title", "name", "artist", "headline"):
        value = event.get(key)
        if _text(value):
            return _text(value)
    return "Concert"


def event_venue(event: Dict[str, Any]) -> str:
    for key in ("venue", "venue_name", "location", "place"):
        value = event.get(key)
        if _text(value):
            return _text(value)
    return "Venue TBD"


def event_city(event: Dict[str, Any]) -> str:
    for key in ("city", "venue_city", "market_city"):
        value = event.get(key)
        if _text(value):
            return _text(value)
    return ""


def event_state(event: Dict[str, Any]) -> str:
    for key in ("state", "venue_state", "region"):
        value = event.get(key)
        if _text(value):
            return _text(value)
    return ""


def event_id(event: Dict[str, Any]) -> str:
    for key in ("event_id", "external_event_id", "source_event_id", "id"):
        value = event.get(key)
        if _text(value):
            return _text(value)
    return f"{_norm(event_title(event))}-{_text(event.get('date'))}-{_norm(event_venue(event))}"


def event_date(event: Dict[str, Any]) -> str:
    for key in ("date", "event_date", "localDate", "start_date"):
        value = event.get(key)
        if _text(value):
            return _text(value)[:10]
    return ""


def event_time(event: Dict[str, Any]) -> str:
    for key in ("time", "event_time", "localTime", "start_time"):
        value = event.get(key)
        if _text(value):
            raw = _text(value)
            if "T" in raw:
                raw = raw.split("T")[-1]
            return raw[:8]
    return ""


def event_score(event: Dict[str, Any]) -> float:
    for key in (
        "final_score", "model_score", "score",
        "recommendation_score", "copilot_score", "rank_score",
    ):
        if event.get(key) not in (None, ""):
            score = _number(event.get(key))
            if 0 < score <= 1:
                score *= 100.0
            return max(0.0, min(score, 99.0))
    return 0.0


def event_url(event: Dict[str, Any]) -> str:
    all_urls = event.get("all_urls") or []
    if isinstance(all_urls, str):
        all_urls = [all_urls]
    for value in list(all_urls) + [
        event.get("url"),
        event.get("ticket_url"),
        event.get("event_url"),
    ]:
        if _text(value).startswith(("http://", "https://")):
            return _text(value)
    return ""


def spotify_url(event: Dict[str, Any]) -> str:
    for key in ("spotify_url", "artist_spotify_url"):
        value = event.get(key)
        if _text(value).startswith(("http://", "https://")):
            return _text(value)
    links = event.get("artist_spotify_urls") or []
    if isinstance(links, list):
        for item in links:
            if isinstance(item, dict) and _text(item.get("url")).startswith(("http://", "https://")):
                return _text(item.get("url"))
    return ""


def known_price(event: Dict[str, Any]) -> float | None:
    for key in ("min_price", "median_price", "average_price"):
        value = event.get(key)
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value)
    return None


def price_text(event: Dict[str, Any]) -> str:
    minimum = event.get("min_price")
    maximum = event.get("max_price")
    median = event.get("median_price")
    average = event.get("average_price")
    source = _text(event.get("price_source"))
    source_suffix = f" · {source}" if source else ""

    if isinstance(minimum, (int, float)) and not pd.isna(minimum):
        if (
            isinstance(maximum, (int, float))
            and not pd.isna(maximum)
            and float(maximum) > float(minimum)
            and float(maximum) <= float(minimum) * 4
        ):
            return f"${float(minimum):.0f}–${float(maximum):.0f}{source_suffix}"
        return f"From ${float(minimum):.0f}{source_suffix}"
    if isinstance(median, (int, float)) and not pd.isna(median):
        return f"Typical ${float(median):.0f}{source_suffix}"
    if isinstance(average, (int, float)) and not pd.isna(average):
        return f"Average ${float(average):.0f}{source_suffix}"
    return ""


def is_direct(event: Dict[str, Any]) -> bool:
    return int(_number(event.get("has_direct_artist_match"))) == 1


def is_weekend(event: Dict[str, Any]) -> bool:
    if int(_number(event.get("weekend_event"))) == 1:
        return True
    parsed = pd.to_datetime(event_date(event), errors="coerce")
    return bool(not pd.isna(parsed) and int(parsed.weekday()) >= 4)


def is_evening(event: Dict[str, Any]) -> bool:
    raw = event_time(event)
    try:
        return int(raw[:2]) >= 17
    except Exception:
        return False


def is_intimate(event: Dict[str, Any]) -> bool:
    venue = event_venue(event).lower()
    return any(word in venue for word in INTIMATE_VENUE_WORDS)


def is_past(event: Dict[str, Any]) -> bool:
    parsed = pd.to_datetime(event_date(event), errors="coerce")
    if pd.isna(parsed):
        return False
    return parsed.date() < date.today()


def event_quality(event: Dict[str, Any]) -> bool:
    title = event_title(event).lower()
    bad_titles = ("parking", "vip upgrade", "club access", "fast lane", "lounge access")
    if any(term in title for term in bad_titles):
        return False
    return bool(event_url(event) or event.get("source"))


def _pick_unique(
    candidates: Sequence[Dict[str, Any]],
    used: set[str],
    minimum_score: float,
    predicate=None,
) -> Dict[str, Any] | None:
    for event in candidates:
        if event_score(event) < minimum_score:
            continue
        if not event_quality(event):
            continue
        if predicate and not predicate(event):
            continue
        key = event_id(event)
        if key in used:
            continue
        used.add(key)
        return event
    return None


def select_featured_picks(events: Sequence[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return only genuinely strong featured recommendations.

    This intentionally returns fewer than five when the candidate pool is weak.
    """
    ranked = sorted(list(events or []), key=event_score, reverse=True)
    used: set[str] = set()
    picks: List[Tuple[str, Dict[str, Any]]] = []

    best = _pick_unique(ranked, used, minimum_score=72)
    if best:
        picks.append(("Best match", best))

    direct = sorted([e for e in ranked if is_direct(e)], key=event_score, reverse=True)
    artist = _pick_unique(direct, used, minimum_score=70)
    if artist:
        picks.append(("Artist you know", artist))

    discovery = sorted(
        [e for e in ranked if not is_direct(e)],
        key=lambda e: (
            event_score(e) * 0.55
            + _number(e.get("discovery_quality_score")) * 0.25
            + _number(e.get("genre_cluster_score")) * 0.20
        ),
        reverse=True,
    )
    discovery_pick = _pick_unique(discovery, used, minimum_score=68)
    if discovery_pick:
        picks.append(("Discovery pick", discovery_pick))

    value_pool = sorted(
        [e for e in ranked if known_price(e) is not None],
        key=lambda e: (
            event_score(e) * 0.70
            + _number(e.get("price_score")) * 0.25
            + min(_number(e.get("source_count"), 1), 3) * 2
        ),
        reverse=True,
    )
    value_pick = _pick_unique(value_pool, used, minimum_score=65)
    if value_pick:
        picks.append(("Best value", value_pick))

    night_pool = sorted(
        ranked,
        key=lambda e: (
            event_score(e) * 0.60
            + _number(e.get("venue_quality_signal")) * 0.20
            + (10 if is_weekend(e) else 0)
            + (7 if is_evening(e) else 0)
        ),
        reverse=True,
    )
    night_pick = _pick_unique(
        night_pool,
        used,
        minimum_score=70,
        predicate=lambda e: is_evening(e) or is_weekend(e),
    )
    if night_pick:
        picks.append(("Best night out", night_pick))

    return picks[:5]


def trust_reasons(event: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []

    if is_direct(event):
        reasons.append("Direct artist match from your Spotify listening")

    artist_rank = max(
        _number(event.get("direct_artist_rank_score")),
        _number(event.get("track_affinity_score")),
        _number(event.get("spotify_durability_score")),
    )
    if artist_rank >= 55 and not is_direct(event):
        reasons.append("Strong artist similarity to music you already play")

    genre_score = _number(event.get("genre_cluster_score"))
    if genre_score >= 55:
        lane = _text(event.get("winning_genre_cluster_label") or event.get("genre"))
        reasons.append(f"Strong fit for your {lane or 'music'} taste")

    discovery_score = max(
        _number(event.get("embedding_rank_score")),
        _number(event.get("discovery_quality_score")),
    )
    if discovery_score >= 55 and not is_direct(event):
        reasons.append("High-confidence discovery based on similar artists and events")

    if is_weekend(event):
        reasons.append("Weekend timing")

    venue_quality = _number(event.get("venue_quality_signal"))
    if venue_quality >= 55:
        reasons.append("Venue and event-quality signals are strong")

    price = price_text(event)
    if price:
        reasons.append(price)

    if not reasons:
        reasons.append("Recommended from your broader listening profile and event similarity")

    return reasons[:5]


def compact_reason(event: Dict[str, Any], max_chars: int = 180) -> str:
    reason = _text(
        event.get("why_artist_match")
        or event.get("why_recommended")
        or event.get("why_taste_lane")
    )
    if not reason:
        reason = trust_reasons(event)[0]
    reason = re.sub(r"\s+", " ", reason).strip()
    if len(reason) > max_chars:
        return reason[: max_chars - 1].rstrip() + "…"
    return reason


def event_status_from_action(action: str) -> str:
    return {
        "want_to_go": "Want",
        "maybe": "Maybe",
        "not_for_me": "Don't go",
    }.get(_text(action), "Saved")


def _ics_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def event_to_ics_lines(event: Dict[str, Any]) -> List[str]:
    raw_date = event_date(event) or date.today().isoformat()
    raw_time = event_time(event) or "19:00:00"
    if len(raw_time) == 5:
        raw_time += ":00"
    try:
        start = datetime.strptime(f"{raw_date[:10]} {raw_time[:8]}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        start = datetime.strptime(raw_date[:10], "%Y-%m-%d").replace(hour=19)
    end = start + timedelta(hours=3)
    location = ", ".join(
        part for part in (
            event_venue(event),
            event_city(event),
            event_state(event),
        ) if part
    )
    description = compact_reason(event, 300)
    url = event_url(event)
    if url:
        description = f"{description}\\nTickets: {url}"

    return [
        "BEGIN:VEVENT",
        f"UID:{_ics_escape(event_id(event) or str(uuid.uuid4()))}@encore-ai",
        f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
        f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}",
        f"SUMMARY:{_ics_escape(event_title(event))}",
        f"LOCATION:{_ics_escape(location)}",
        f"DESCRIPTION:{_ics_escape(description)}",
        "END:VEVENT",
    ]


def build_playlist_ics(events: Iterable[Dict[str, Any]]) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        "PRODID:-//Encore AI//My Shows//EN",
        "METHOD:PUBLISH",
    ]
    for event in events:
        lines.extend(event_to_ics_lines(event))
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def _coordinate(event: Dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = event.get(key)
        try:
            if value not in (None, ""):
                return float(value)
        except Exception:
            continue
    raw = event.get("raw_json") or event.get("raw") or {}
    if isinstance(raw, dict):
        venue = raw.get("venue") or {}
        if isinstance(venue, dict):
            for key in keys:
                try:
                    value = venue.get(key)
                    if value not in (None, ""):
                        return float(value)
                except Exception:
                    continue
        embedded = raw.get("_embedded") or {}
        venues = embedded.get("venues") if isinstance(embedded, dict) else None
        if isinstance(venues, list) and venues:
            location = venues[0].get("location") or {}
            if isinstance(location, dict):
                for key in keys:
                    try:
                        value = location.get(key)
                        if value not in (None, ""):
                            return float(value)
                    except Exception:
                        continue
    return None


def build_map_dataframe(events: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for event in events:
        latitude = _coordinate(event, ("latitude", "lat", "venue_latitude", "venue_lat"))
        longitude = _coordinate(event, ("longitude", "lon", "lng", "venue_longitude", "venue_lon"))
        if latitude is None or longitude is None:
            continue
        rows.append({
            "lat": latitude,
            "lon": longitude,
            "show": event_title(event),
            "venue": event_venue(event),
            "city": event_city(event),
            "match": round(event_score(event), 1),
            "date": event_date(event),
            "price": price_text(event) or "Price unavailable",
        })
    return pd.DataFrame(rows)


def render_event_map(events: Sequence[Dict[str, Any]]) -> bool:
    frame = build_map_dataframe(events)
    if frame.empty:
        st.info("Map view is unavailable for these results because the event sources did not return venue coordinates.")
        return False

    figure = px.scatter_mapbox(
        frame,
        lat="lat",
        lon="lon",
        hover_name="show",
        hover_data={
            "venue": True,
            "city": True,
            "match": True,
            "date": True,
            "price": True,
            "lat": False,
            "lon": False,
        },
        zoom=9,
        height=620,
    )
    figure.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    return True
