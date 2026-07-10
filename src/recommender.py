from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .features import build_user_profile_text, build_event_text, make_feature_row, feature_frame
from .genre_clusters import build_user_taste_clusters
from .embeddings import embed_texts
from .vector_store import upsert_events_to_chroma
from .modeling import predict_feedback_scores, load_feedback_model


def _normalize_0_100(series):
    values = pd.Series(series, dtype=float)
    if len(values) == 0:
        return []
    if values.max() == values.min():
        return [50.0 for _ in values]
    return (((values - values.min()) / (values.max() - values.min())) * 100.0).tolist()


def _tracks_for_artists(top_tracks, artist_names, limit=2):
    names = {str(name).lower() for name in artist_names if name}
    output = []
    for track in top_tracks:
        if str(track.get("artist") or "").lower() in names and track.get("track"):
            output.append(track.get("track"))
        if len(output) >= limit:
            break
    return output


def _confidence_label(features):
    if int(features.get("has_direct_artist_match") or 0) == 1:
        return "Direct match"
    cluster_score = float(features.get("genre_cluster_score") or 0)
    embedding = float(features.get("embedding_rank_score") or 0)
    if cluster_score >= 55 and embedding >= 65:
        return "Strong match"
    if cluster_score >= 35 or embedding >= 78:
        return "Relevant discovery"
    return "Exploratory"



def spotify_links_for_event(top_artists, event):
    lookup = {
        str(artist.get("artist") or "").lower(): artist.get("spotify_url")
        for artist in top_artists
        if artist.get("artist") and artist.get("spotify_url")
    }
    links = []
    for artist in event.get("artists") or []:
        url = lookup.get(str(artist).lower())
        if url and url not in [x.get("url") for x in links]:
            links.append({"artist": artist, "url": url})
    return links[:3]


def build_reason_tags(features, event):
    tags = []
    if int(features.get("has_direct_artist_match") or 0):
        tags.append("Direct match")
    if float(features.get("direct_artist_rank_score") or 0) >= 60:
        tags.append("Recent listening")
    if float(features.get("spotify_durability_score") or 0) >= 65:
        tags.append("Long-term fit")
    if not int(features.get("has_direct_artist_match") or 0) and float(features.get("discovery_quality_score") or 0) >= 45:
        tags.append("Strong discovery")
    if float(features.get("venue_quality_signal") or 0) >= 50:
        tags.append("Strong venue")
    if int(features.get("weekend_event") or 0):
        tags.append("Weekend")
    lane = features.get("winning_genre_cluster_label")
    if lane:
        tags.append(lane)
    return tags[:5]

def build_reason_details(event, features, top_tracks):
    direct = features.get("direct_artist_matches") or []
    anchors = features.get("anchor_artists") or []
    cluster_label = features.get("winning_genre_cluster_label") or "your broader taste"
    cluster_score = float(features.get("genre_cluster_score") or 0)
    embedding_rank = float(features.get("embedding_rank_score") or 0)

    if direct:
        tracks = _tracks_for_artists(top_tracks, direct)
        artist_reason = f"You already listen to {', '.join(direct[:2])}."
        if tracks:
            artist_reason += f" Your Spotify history includes {', '.join([f'“{t}”' for t in tracks])}."
    elif anchors:
        artist_reason = (
            f"This fits your {cluster_label} lane, where your strongest related artists include "
            f"{', '.join(anchors[:3])}."
        )
    elif features.get("winning_genre_cluster"):
        artist_reason = f"This aligns with your {cluster_label} listening cluster."
    else:
        artist_reason = "This is an exploratory pick based on the event description and your broader listening profile."

    if features.get("winning_genre_cluster"):
        lane_reason = f"This fits your {cluster_label} taste cluster."
    else:
        lane_reason = "Genre metadata was limited, so this recommendation relies more on overall event similarity."

    signals = []
    if int(features.get("has_direct_artist_match") or 0):
        signals.append("direct Spotify artist match")
    if embedding_rank >= 80:
        signals.append("strong overall similarity")
    elif embedding_rank >= 60:
        signals.append("solid overall similarity")
    if int(features.get("weekend_event") or 0):
        signals.append("weekend show")
    if int(features.get("known_price") or 0):
        price = float(features.get("min_price_filled") or 0)
        signals.append(f"price available around ${price:.0f}")
    else:
        signals.append("live price not published by connected sources")
    if int(features.get("has_multiple_sources") or 0):
        signals.append("confirmed by multiple event sources")

    confidence = _confidence_label(features)
    return {
        "artist_match": artist_reason,
        "taste_lane": lane_reason,
        "timing_value": "; ".join(signals).capitalize() + ".",
        "confidence": confidence,
        "summary": f"{artist_reason} {lane_reason} {'; '.join(signals).capitalize()}.",
    }


def _score_with_mode(feature_df: pd.DataFrame, recommendation_mode: str) -> pd.Series:
    """Cold-start score by product-facing recommendation style."""
    cluster_gate = (0.25 + 0.75 * (feature_df["genre_cluster_score"].clip(0, 100) / 100.0))
    direct_gate = feature_df["has_direct_artist_match"].clip(0, 1)
    effective_embedding = feature_df["embedding_rank_score"] * cluster_gate
    effective_embedding = effective_embedding.where(direct_gate == 0, feature_df["embedding_rank_score"])

    mode = (recommendation_mode or "Best overall").lower()

    if "familiar" in mode:
        return (
            0.43 * feature_df["exact_norm"]
            + 0.15 * feature_df["cluster_norm"]
            + 0.11 * effective_embedding
            + 0.10 * feature_df["durability_norm"]
            + 0.08 * feature_df["track_norm"]
            + 0.05 * feature_df["venue_quality_signal"].clip(0, 100)
            + 0.04 * feature_df["price_norm"]
            + 0.03 * feature_df["days_norm"]
            + 0.01 * feature_df["source_count_score"]
            - (1.0 - direct_gate) * 6.0
        ).clip(lower=0)

    if "up" in mode:
        # Emerging but not random: needs taste fit + quality signals.
        return (
            0.08 * feature_df["exact_norm"]
            + 0.28 * feature_df["cluster_norm"]
            + 0.22 * effective_embedding
            + 0.20 * feature_df["discovery_quality_score"].clip(0, 100)
            + 0.08 * feature_df["venue_quality_signal"].clip(0, 100)
            + 0.05 * feature_df["source_count_score"]
            + 0.04 * feature_df["price_norm"]
            + 0.03 * feature_df["days_norm"]
            + 0.02 * feature_df["weekend_event"] * 100
            - direct_gate * 4.0
        ).clip(lower=0)

    if "discover" in mode or "fresh" in mode:
        return (
            0.12 * feature_df["exact_norm"]
            + 0.31 * feature_df["cluster_norm"]
            + 0.25 * effective_embedding
            + 0.12 * feature_df["discovery_quality_score"].clip(0, 100)
            + 0.07 * feature_df["novelty_score"]
            + 0.05 * feature_df["venue_quality_signal"].clip(0, 100)
            + 0.04 * feature_df["price_norm"]
            + 0.03 * feature_df["days_norm"]
            + 0.01 * feature_df["weekend_event"] * 100
        ).clip(lower=0)

    # Best overall: balanced, reliable, and explainable.
    return (
        0.29 * feature_df["exact_norm"]
        + 0.25 * feature_df["cluster_norm"]
        + 0.18 * effective_embedding
        + 0.06 * feature_df["durability_norm"]
        + 0.06 * feature_df["track_norm"]
        + 0.05 * feature_df["discovery_quality_score"].clip(0, 100)
        + 0.04 * feature_df["venue_quality_signal"].clip(0, 100)
        + 0.04 * feature_df["price_norm"]
        + 0.02 * feature_df["days_norm"]
        + 0.01 * feature_df["weekend_event"] * 100
    ).clip(lower=0)


def rank_events_v6(
    top_artists: List[Dict[str, Any]],
    top_tracks: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    use_trained_model: bool = True,
    recommendation_mode: str = "Balanced",
    model_variant: str = "current",
    model_weight: float | None = None,
):
    if not events:
        return []

    taste_clusters = build_user_taste_clusters(top_artists, top_tracks)
    user_text = build_user_profile_text(top_artists, top_tracks)
    event_texts = [build_event_text(event) for event in events]
    all_embeddings = embed_texts([user_text] + event_texts)
    user_embedding = all_embeddings[0]
    event_embeddings = all_embeddings[1:]
    upsert_events_to_chroma(events, event_texts, event_embeddings)

    similarities = cosine_similarity(user_embedding.reshape(1, -1), event_embeddings).flatten()
    feature_rows = [
        make_feature_row(top_artists, top_tracks, event, similarity, taste_clusters=taste_clusters)
        for event, similarity in zip(events, similarities)
    ]
    features = feature_frame(feature_rows)

    features["embedding_rank_score"] = features["embedding_similarity"].rank(pct=True).fillna(0.5) * 100.0
    features["exact_norm"] = _normalize_0_100(features["exact_artist_score"])
    features["cluster_norm"] = _normalize_0_100(features["genre_cluster_score"])
    features["price_norm"] = _normalize_0_100(features["price_score"])
    features["days_norm"] = _normalize_0_100(features["days_score"])
    features["durability_norm"] = _normalize_0_100(features["spotify_durability_score"])
    features["track_norm"] = _normalize_0_100(features["track_affinity_score"])
    features["hybrid_score"] = _score_with_mode(features, recommendation_mode)

    for idx, row in enumerate(feature_rows):
        for column in features.columns:
            if column in row or column in {
                "embedding_rank_score", "hybrid_score", "source_count_score",
                "artist_popularity_signal", "genre_cluster_score",
            }:
                value = features.iloc[idx][column]
                if isinstance(value, (int, float)):
                    row[column] = float(value)
        row["match_confidence"] = _confidence_label(row)
        row["recommendation_mode"] = recommendation_mode

    bundle = load_feedback_model(model_variant) if use_trained_model else None
    has_model = bundle is not None
    if has_model:
        model_scores = predict_feedback_scores(features, model_variant)
        score_source = f"{model_variant}_learning_to_rank"
        if model_weight is None:
            model_weight = float(bundle.get("recommended_model_weight") or 0.20)
    else:
        model_scores = features["hybrid_score"].values
        score_source = "spotify_genre_cluster_baseline"
        model_weight = 0.0

    model_weight = max(0.0, min(float(model_weight or 0.0), 0.45))
    ranked = []
    for event, feature_row, model_score in zip(events, feature_rows, model_scores):
        base_score = float(feature_row["hybrid_score"])
        final_score = (1.0 - model_weight) * base_score + model_weight * float(model_score) if has_model else base_score
        reason = build_reason_details(event, feature_row, top_tracks)
        ranked.append({
            **event,
            **feature_row,
            "model_score": round(float(model_score), 2),
            "model_weight": model_weight,
            "final_score": round(float(final_score), 2),
            "score_source": score_source,
            "why_recommended": reason["summary"],
            "why_artist_match": reason["artist_match"],
            "why_taste_lane": reason["taste_lane"],
            "why_timing_value": reason["timing_value"],
            "why_confidence": reason["confidence"],
            "reason_tags": build_reason_tags(feature_row, event),
            "artist_spotify_urls": spotify_links_for_event(top_artists, event),
        })
    return sorted(ranked, key=lambda item: item["final_score"], reverse=True)
