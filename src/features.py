from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
import math
import pandas as pd

from .genre_clusters import (
    CLUSTER_KEYS,
    anchors_for_cluster,
    build_user_taste_clusters,
    classify_event_cluster,
    event_cluster_feature_values,
)


def normalize_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(str(v).lower() for v in value if v)
    return str(value).lower()


def build_user_profile_text(top_artists: List[Dict[str, Any]], top_tracks: List[Dict[str, Any]]) -> str:
    taste_clusters = build_user_taste_clusters(top_artists, top_tracks)
    parts: List[str] = []
    for artist in top_artists:
        rank = max(1, int(artist.get("rank") or 50))
        repeats = max(1, int(8 / math.sqrt(rank)))
        parts.extend([artist.get("artist", "")] * repeats)
        parts.extend((artist.get("genres") or []) * max(1, repeats // 2))
    for track in top_tracks:
        if track.get("track"):
            parts.append(track.get("track"))
        if track.get("artist"):
            parts.append(track.get("artist"))
    for cluster_key, score in taste_clusters.get("dominant_clusters", [])[:5]:
        parts.extend([cluster_key.replace("_", " ")] * max(1, int(score / 20)))
    return normalize_text(parts)


def build_event_text(event: Dict[str, Any]) -> str:
    parts = [
        event.get("event_name", ""),
        event.get("venue", ""),
        event.get("city", ""),
        event.get("genre", ""),
        event.get("subgenre", ""),
        event.get("segment", ""),
    ]
    parts.extend(event.get("artists", []) or [])
    return normalize_text(parts)


def _artist_lookup(top_artists: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(artist.get("artist") or "").lower(): artist
        for artist in top_artists
        if artist.get("artist")
    }


def _track_artist_lookup(top_tracks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for track in top_tracks:
        artist = str(track.get("artist") or "").lower()
        if artist:
            lookup.setdefault(artist, []).append(track)
    return lookup


def _compact_artist_name(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _artist_name_matches(artist_name: str, event_text: str) -> bool:
    artist_key = _compact_artist_name(artist_name)
    event_key = _compact_artist_name(event_text)
    if not artist_key or len(artist_key) < 4 or not event_key:
        return False
    return artist_key in event_key


def exact_artist_match_features(top_artists, top_tracks, event):
    artist_lookup = _artist_lookup(top_artists)
    track_lookup = _track_artist_lookup(top_tracks)
    event_artists = [artist for artist in event.get("artists", []) or [] if artist]

    event_text = " ".join([
        str(event.get("event_name") or ""),
        str(event.get("title") or ""),
        str(event.get("name") or ""),
        " ".join(str(a) for a in event_artists),
    ])

    direct_matches: List[str] = []
    track_artist_matches: List[str] = []
    direct_artist_rank_score = 0.0
    spotify_durability_score = 0.0
    track_affinity_score = 0.0
    artist_blend_score = 0.0

    def add_direct_match(display_name, artist_obj):
        nonlocal direct_artist_rank_score, spotify_durability_score, artist_blend_score
        if display_name and display_name not in direct_matches:
            direct_matches.append(display_name)
        rank = max(1, int(artist_obj.get("rank") or 50))
        direct_artist_rank_score = max(direct_artist_rank_score, max(0.0, 115.0 - rank * 2.4))
        time_ranges = set(artist_obj.get("time_ranges") or [])
        spotify_durability_score = max(spotify_durability_score, len(time_ranges) / 3.0 * 100.0)
        artist_blend_score = max(artist_blend_score, float(artist_obj.get("blend_score") or 0.0))

    for event_artist in event_artists:
        key = str(event_artist or "").lower()
        artist = artist_lookup.get(key)
        if artist:
            add_direct_match(artist.get("artist") or event_artist, artist)

        tracks = track_lookup.get(key, [])
        if tracks:
            if event_artist not in track_artist_matches:
                track_artist_matches.append(event_artist)
            best_rank = min(max(1, int(track.get("rank") or 50)) for track in tracks)
            track_affinity_score = max(track_affinity_score, max(0.0, 110.0 - best_rank * 2.0))

    for artist in top_artists:
        name = artist.get("artist")
        if name and name not in direct_matches and _artist_name_matches(name, event_text):
            add_direct_match(name, artist)

    for artist_name, tracks in track_lookup.items():
        if artist_name and _artist_name_matches(artist_name, event_text):
            display = tracks[0].get("artist") or artist_name
            if display not in track_artist_matches:
                track_artist_matches.append(display)
            best_rank = min(max(1, int(track.get("rank") or 50)) for track in tracks)
            track_affinity_score = max(track_affinity_score, max(0.0, 110.0 - best_rank * 2.0))

    all_matches: List[str] = []
    for name in direct_matches + track_artist_matches:
        if name and name not in all_matches:
            all_matches.append(name)

    exact_artist_score = (
        direct_artist_rank_score * 0.70
        + track_affinity_score * 0.18
        + spotify_durability_score * 0.12
    )
    return {
        "exact_artist_score": min(exact_artist_score, 100.0),
        "direct_artist_matches": all_matches,
        "top_artist_matches": direct_matches,
        "top_track_artist_matches": track_artist_matches,
        "has_direct_artist_match": int(bool(all_matches)),
        "direct_artist_rank_score": direct_artist_rank_score,
        "track_affinity_score": track_affinity_score,
        "spotify_durability_score": spotify_durability_score,
        "artist_blend_score": artist_blend_score,
    }


def artist_popularity_signal(top_artists, event):
    popularity = {
        str(artist.get("artist") or "").lower(): artist.get("popularity")
        for artist in top_artists
        if artist.get("artist")
    }
    values = []
    for artist in event.get("artists", []) or []:
        value = popularity.get(str(artist).lower())
        if isinstance(value, (int, float)):
            values.append(float(value))
    return max(values) if values else 0.0


def event_time_features(event):
    date_str = event.get("date")
    time_str = event.get("time")
    values = {
        "days_until_event": 0.0,
        "weekend_event": 0.0,
        "friday_event": 0.0,
        "saturday_event": 0.0,
        "weekday_index": 0.0,
        "event_hour": 19.0,
        "evening_event": 1.0,
        "known_event_time": 0.0,
    }
    if date_str:
        try:
            event_date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
            today = datetime.now().date()
            weekday = event_date.weekday()
            values.update({
                "days_until_event": float(max((event_date - today).days, 0)),
                "weekend_event": float(weekday in [4, 5, 6]),
                "friday_event": float(weekday == 4),
                "saturday_event": float(weekday == 5),
                "weekday_index": float(weekday),
            })
        except Exception:
            pass
    if time_str:
        for pattern in ("%H:%M:%S", "%H:%M"):
            try:
                parsed = datetime.strptime(str(time_str)[:8] if pattern == "%H:%M:%S" else str(time_str)[:5], pattern)
                values["event_hour"] = float(parsed.hour + parsed.minute / 60.0)
                values["evening_event"] = float(parsed.hour >= 17)
                values["known_event_time"] = 1.0
                break
            except Exception:
                continue
    return values




def venue_quality_signal(event):
    """Lightweight event-quality proxy from venue/source metadata.

    This is intentionally deterministic and explainable for a portfolio product:
    recognizable larger venues, multi-source confirmation, pricing, images, and
    listing counts all increase quality. Unknown local events can still rank if
    taste fit is strong, but they will not dominate merely because text matches.
    """
    venue = str(event.get("venue") or "").lower()
    strong_venues = [
        "moody center", "acl live", "austin city limits", "stubb", "mohawk",
        "emo's", "emos", "scoot inn", "antone", "the parish", "empire control",
        "germania", "bass concert", "madison square garden", "red rocks",
        "rady shell", "greek theatre", "hollywood bowl", "ryman", "brooklyn steel",
        "9:30 club", "troubadour", "fillmore", "house of blues", "terminal 5",
    ]
    score = 18.0
    if any(name in venue for name in strong_venues):
        score += 34.0
    if int(event.get("source_count") or len(event.get("sources") or []) or 1) > 1:
        score += 18.0
    if event.get("image_url"):
        score += 8.0
    if isinstance(event.get("min_price"), (int, float)) or isinstance(event.get("median_price"), (int, float)):
        score += 10.0
    listings = float(event.get("listing_count") or 0.0)
    if listings:
        score += min(20.0, listings / 5.0)
    return min(score, 100.0)


def _popularity_bucket(value):
    try:
        value = float(value or 0)
    except Exception:
        value = 0.0
    return value

def price_features(event):
    min_price = event.get("min_price")
    median_price = event.get("median_price")
    average_price = event.get("average_price")
    display_price = min_price if isinstance(min_price, (int, float)) else median_price
    if not isinstance(display_price, (int, float)):
        display_price = average_price
    known_price = int(isinstance(display_price, (int, float)) and display_price > 0)
    price_value = float(display_price) if known_price else 90.0
    affordability_score = max(0.0, 100.0 - min(price_value, 300.0) / 3.0) if known_price else 35.0
    return {
        "known_price": float(known_price),
        "known_starting_price": float(isinstance(min_price, (int, float)) and min_price > 0),
        "known_typical_price": float(
            isinstance(median_price, (int, float)) or isinstance(average_price, (int, float))
        ),
        "min_price_filled": price_value,
        "price_score": affordability_score,
        "price_under_50": float(known_price and price_value <= 50),
        "price_50_100": float(known_price and 50 < price_value <= 100),
        "price_100_175": float(known_price and 100 < price_value <= 175),
        "price_over_175": float(known_price and price_value > 175),
    }


def make_feature_row(
    top_artists,
    top_tracks,
    event,
    embedding_similarity,
    taste_clusters: Dict[str, Any] | None = None,
):
    taste_clusters = taste_clusters or build_user_taste_clusters(top_artists, top_tracks)
    exact = exact_artist_match_features(top_artists, top_tracks, event)
    cluster = classify_event_cluster(
        event,
        direct_matches=exact.get("direct_artist_matches") or [],
        taste_clusters=taste_clusters,
    )
    cluster_key = cluster.get("winning_genre_cluster")
    user_cluster_affinity = float((taste_clusters.get("scores") or {}).get(cluster_key, 0.0)) if cluster_key else 0.0
    event_cluster_confidence = float(cluster.get("event_cluster_confidence") or 0.0)
    genre_cluster_score = user_cluster_affinity * (event_cluster_confidence / 100.0)
    anchors = anchors_for_cluster(
        taste_clusters,
        cluster_key,
        direct_matches=exact.get("direct_artist_matches") or [],
        limit=3,
    )

    timing = event_time_features(event)
    price = price_features(event)
    venue_quality = venue_quality_signal(event)
    days = timing["days_until_event"]
    days_score = max(0.0, 100.0 - min(days, 240.0) / 2.4)
    source_count = int(event.get("source_count") or len(event.get("sources") or []) or 1)
    source_count_score = min(source_count / 3.0 * 100.0, 100.0)
    listing_count = float(event.get("listing_count") or 0.0)
    listing_count_signal = min(math.log1p(max(0.0, listing_count)) / math.log(101) * 100.0, 100.0)
    popularity_signal = artist_popularity_signal(top_artists, event)
    familiarity_score = min(
        100.0,
        exact["direct_artist_rank_score"] * 0.55
        + exact["track_affinity_score"] * 0.25
        + exact["spotify_durability_score"] * 0.20,
    )
    novelty_score = max(0.0, 100.0 - familiarity_score)
    discovery_quality_score = min(
        100.0,
        genre_cluster_score * 0.36
        + venue_quality * 0.22
        + source_count_score * 0.15
        + price["known_price"] * 10.0
        + min(float(event.get("listing_count") or 0.0), 80.0) * 0.10
        + (1.0 if event.get("image_url") else 0.0) * 5.0
        + min(popularity_signal, 100.0) * 0.12,
    )

    row = {
        **exact,
        **timing,
        **price,
        **event_cluster_feature_values(cluster_key),
        "winning_genre_cluster": cluster_key,
        "winning_genre_cluster_label": cluster.get("winning_genre_cluster_label"),
        "anchor_artists": anchors,
        "user_cluster_affinity": user_cluster_affinity,
        "event_cluster_confidence": event_cluster_confidence,
        "genre_cluster_score": genre_cluster_score,
        # Backwards-compatible names used by old databases and cards.
        "genre_overlap_count": 1.0 if cluster_key and genre_cluster_score > 0 else 0.0,
        "genre_score": genre_cluster_score,
        "matched_genres": [cluster.get("winning_genre_cluster_label")] if cluster_key else [],
        "embedding_similarity": float(embedding_similarity),
        "embedding_score": float(embedding_similarity) * 100.0,
        "days_score": days_score,
        "source_count_score": source_count_score,
        "has_multiple_sources": float(source_count > 1),
        "listing_count_signal": listing_count_signal,
        "artist_popularity_signal": popularity_signal,
        "venue_quality_signal": venue_quality,
        "discovery_quality_score": discovery_quality_score,
        "familiarity_score": familiarity_score,
        "novelty_score": novelty_score,
        "hybrid_score": 0.0,
    }
    return row


BASE_NUMERIC_COLS = [
    "exact_artist_score", "has_direct_artist_match", "direct_artist_rank_score",
    "track_affinity_score", "spotify_durability_score", "artist_blend_score",
    "genre_overlap_count", "genre_score", "user_cluster_affinity",
    "event_cluster_confidence", "genre_cluster_score", "known_price",
    "known_starting_price", "known_typical_price", "min_price_filled", "price_score",
    "price_under_50", "price_50_100", "price_100_175", "price_over_175",
    "weekend_event", "friday_event", "saturday_event", "weekday_index",
    "event_hour", "evening_event", "known_event_time", "days_until_event", "days_score",
    "embedding_similarity", "embedding_score", "embedding_rank_score",
    "source_count_score", "has_multiple_sources", "listing_count_signal",
    "artist_popularity_signal", "venue_quality_signal", "discovery_quality_score",
    "familiarity_score", "novelty_score", "hybrid_score",
] + [f"cluster_{key}" for key in CLUSTER_KEYS]


def feature_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in BASE_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df
