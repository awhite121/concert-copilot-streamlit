from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Set
import json
import pandas as pd

from .database import (
    insert_interaction,
    load_interactions,
    load_latest_preferences,
    load_user_latest_preferences,
    load_user_opened_tickets,
    delete_existing_preference,
    delete_preference,
)
from .genre_clusters import CLUSTER_KEYS

ACTION_LABELS = {
    "want_to_go": 3,
    "maybe": 2,
    "not_for_me": 0,
    "opened_tickets": None,
    "shown": None,
}

PREFERENCE_ACTIONS = {"want_to_go", "maybe", "not_for_me"}


def event_to_interaction_row(
    user: Dict[str, Any],
    session_id: str,
    event: Dict[str, Any],
    action: str,
    rank_position: int = None,
    feedback_reason: str | None = None,
    feedback_reasons: List[str] | None = None,
) -> Dict[str, Any]:
    label = ACTION_LABELS.get(action)
    row = {
        "created_at": datetime.utcnow().isoformat(),
        "user_id": user.get("user_id", "unknown_user"),
        "user_display_name": user.get("display_name"),
        "session_id": session_id,
        "rank_position": rank_position,
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "event_date": event.get("date"),
        "event_time": event.get("time"),
        "venue": event.get("venue"),
        "city": event.get("city"),
        "state": event.get("state"),
        "genre": event.get("genre"),
        "subgenre": event.get("subgenre"),
        "artists_json": json.dumps(event.get("artists", [])),
        "source": event.get("source"),
        "sources_json": json.dumps(event.get("sources", [])),
        "url": event.get("url"),
        "image_url": event.get("image_url"),
        "min_price": event.get("min_price"),
        "max_price": event.get("max_price"),
        "price_source": event.get("price_source"),
        "match_confidence": event.get("match_confidence"),
        "why_recommended": event.get("why_recommended"),
        "feedback_reason": "; ".join(feedback_reasons) if feedback_reasons else feedback_reason,
        "action": action,
        "label": label,
        "winning_genre_cluster": event.get("winning_genre_cluster"),
        "winning_genre_cluster_label": event.get("winning_genre_cluster_label"),
        "anchor_artists_json": json.dumps(event.get("anchor_artists", [])),
    }

    feature_names = [
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
        "familiarity_score", "novelty_score",
        "hybrid_score", "model_score", "final_score",
    ] + [f"cluster_{key}" for key in CLUSTER_KEYS]
    for feature in feature_names:
        row[feature] = event.get(feature)
    return row


def save_feedback_action(
    user: Dict[str, Any],
    session_id: str,
    event: Dict[str, Any],
    action: str,
    rank_position: int = None,
    feedback_reason: str | None = None,
    feedback_reasons: List[str] | None = None,
):
    if action in PREFERENCE_ACTIONS:
        delete_existing_preference(
            user_id=user.get("user_id", "unknown_user"),
            event_id=event.get("event_id"),
        )
    row = event_to_interaction_row(
        user,
        session_id,
        event,
        action,
        rank_position=rank_position,
        feedback_reason=feedback_reason,
        feedback_reasons=feedback_reasons,
    )
    insert_interaction(row)
    return row


def clear_feedback_preference(user_id: str, event_id: str):
    delete_preference(user_id, event_id)


def log_impressions_once(user: Dict[str, Any], session_id: str, ranked_events: List[Dict[str, Any]], top_n: int = 25):
    for idx, event in enumerate(ranked_events[:top_n], start=1):
        save_feedback_action(user, session_id, event, "shown", rank_position=idx)


def get_feedback_df() -> pd.DataFrame:
    return load_interactions(include_shown=True)


def get_labeled_feedback_df() -> pd.DataFrame:
    return load_latest_preferences()


def get_user_rated_event_ids(user_id: str) -> Set[str]:
    df = load_user_latest_preferences(user_id)
    if df.empty or "event_id" not in df.columns:
        return set()
    return set(df["event_id"].dropna().astype(str).tolist())


def get_user_not_for_me_event_ids(user_id: str) -> Set[str]:
    df = load_user_latest_preferences(user_id)
    if df.empty:
        return set()
    df = df[df["action"] == "not_for_me"]
    return set(df["event_id"].dropna().astype(str).tolist())


def get_user_shortlist_df(user_id: str) -> pd.DataFrame:
    prefs = load_user_latest_preferences(user_id)
    if prefs.empty:
        return pd.DataFrame()
    prefs = prefs[prefs["action"].isin(["want_to_go", "maybe"])].copy()
    if prefs.empty:
        return prefs
    return prefs.sort_values(["action", "created_at"], ascending=[True, False])


def interaction_row_to_event(row: Dict[str, Any]) -> Dict[str, Any]:
    def _loads(value, default):
        try:
            return json.loads(value) if value else default
        except Exception:
            return default

    event = {
        "event_id": row.get("event_id"),
        "event_name": row.get("event_name"),
        "date": row.get("event_date"),
        "time": row.get("event_time"),
        "venue": row.get("venue"),
        "city": row.get("city"),
        "state": row.get("state"),
        "genre": row.get("genre"),
        "subgenre": row.get("subgenre"),
        "artists": _loads(row.get("artists_json"), []),
        "source": row.get("source"),
        "sources": _loads(row.get("sources_json"), []),
        "url": row.get("url"),
        "image_url": row.get("image_url"),
        "min_price": row.get("min_price"),
        "max_price": row.get("max_price"),
        "price_source": row.get("price_source"),
        "match_confidence": row.get("match_confidence"),
        "why_recommended": row.get("why_recommended"),
        "winning_genre_cluster": row.get("winning_genre_cluster"),
        "winning_genre_cluster_label": row.get("winning_genre_cluster_label"),
        "anchor_artists": _loads(row.get("anchor_artists_json"), []),
    }
    for feature in [
        "exact_artist_score", "has_direct_artist_match", "genre_score",
        "genre_cluster_score", "embedding_rank_score", "hybrid_score", "model_score", "final_score",
    ]:
        event[feature] = row.get(feature)
    return event


def summarize_feedback() -> Dict[str, Any]:
    df = get_feedback_df()
    latest = get_labeled_feedback_df()
    if df.empty:
        return {
            "total_rows": 0,
            "labeled_rows": 0,
            "latest_preference_rows": 0,
            "unique_users": 0,
            "unique_sessions": 0,
            "actions": {},
            "reasons": {},
        }
    labeled = df[df["label"].notna()]
    reasons = {}
    if "feedback_reason" in latest.columns:
        reasons = latest["feedback_reason"].dropna().value_counts().to_dict()
    return {
        "total_rows": len(df),
        "labeled_rows": len(labeled),
        "latest_preference_rows": len(latest),
        "unique_users": df["user_id"].nunique(),
        "unique_sessions": df["session_id"].nunique(),
        "actions": df["action"].value_counts().to_dict(),
        "reasons": reasons,
    }
