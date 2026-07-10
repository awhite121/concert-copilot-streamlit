from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import math
import re


CLUSTERS: Dict[str, Dict[str, Any]] = {
    "country_americana": {
        "label": "Country / Americana",
        "aliases": [
            "country", "americana", "red dirt", "texas country", "alt country",
            "outlaw country", "contemporary country", "country pop", "country rock",
            "southern rock", "roots country", "honky tonk", "bluegrass",
        ],
    },
    "folk_singer_songwriter": {
        "label": "Folk / Singer-Songwriter",
        "aliases": [
            "folk", "folk pop", "folk rock", "singer songwriter", "acoustic",
            "indie folk", "roots", "chamber folk", "stomp and holler",
        ],
    },
    "hip_hop_rap": {
        "label": "Hip-Hop / Rap",
        "aliases": [
            "hip hop", "hip-hop", "rap", "trap", "drill", "melodic rap",
            "pop rap", "southern hip hop", "cloud rap", "gangster rap",
        ],
    },
    "electronic_dance": {
        "label": "Electronic / Dance",
        "aliases": [
            "electronic", "edm", "house", "techno", "dance", "electro",
            "trance", "dubstep", "drum and bass", "indietronica", "future bass",
            "deep house", "progressive house", "electronica",
        ],
    },
    "indie_alternative": {
        "label": "Indie / Alternative",
        "aliases": [
            "indie", "alternative", "alt rock", "indie rock", "indie pop",
            "dream pop", "shoegaze", "post punk", "psychedelic pop",
            "psychedelic rock", "art pop", "bedroom pop",
        ],
    },
    "rock_punk_metal": {
        "label": "Rock / Punk / Metal",
        "aliases": [
            "rock", "hard rock", "classic rock", "punk", "pop punk", "metal",
            "metalcore", "heavy metal", "garage rock", "emo", "grunge",
        ],
    },
    "pop": {
        "label": "Pop",
        "aliases": [
            "pop", "dance pop", "electropop", "synthpop", "teen pop",
            "pop rock", "art pop", "indie pop", "hyperpop",
        ],
    },
    "rnb_soul_funk": {
        "label": "R&B / Soul / Funk",
        "aliases": [
            "r&b", "rnb", "soul", "neo soul", "funk", "motown",
            "alternative r&b", "contemporary r&b",
        ],
    },
    "latin": {
        "label": "Latin",
        "aliases": [
            "latin", "reggaeton", "latin pop", "salsa", "bachata", "cumbia",
            "regional mexican", "corridos", "mariachi", "urbano latino",
        ],
    },
    "jazz_blues": {
        "label": "Jazz / Blues",
        "aliases": ["jazz", "blues", "smooth jazz", "bebop", "delta blues", "swing"],
    },
    "classical_instrumental": {
        "label": "Classical / Instrumental",
        "aliases": [
            "classical", "orchestra", "symphony", "instrumental", "ambient",
            "neo classical", "film score", "piano",
        ],
    },
}

CLUSTER_KEYS = list(CLUSTERS.keys())


def _normalize(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cluster_label(key: str | None) -> str:
    if not key:
        return "Unclear taste lane"
    return CLUSTERS.get(key, {}).get("label", key.replace("_", " ").title())


def match_cluster_scores(text: Any) -> Dict[str, float]:
    normalized = f" {_normalize(text)} "
    scores: Dict[str, float] = {}
    for key, spec in CLUSTERS.items():
        score = 0.0
        for alias in spec["aliases"]:
            alias_norm = _normalize(alias)
            if not alias_norm:
                continue
            # Exact phrase matches are much safer than loose token overlap.
            if f" {alias_norm} " in normalized:
                score += 2.5 + min(len(alias_norm.split()), 3) * 0.6
            elif len(alias_norm) >= 6 and alias_norm in normalized:
                score += 1.0
        if score > 0:
            scores[key] = score
    return scores


def artist_weight(artist: Dict[str, Any]) -> float:
    rank = max(1, int(artist.get("rank") or 50))
    blend_score = float(artist.get("blend_score") or max(1, 51 - rank))
    durability = len(set(artist.get("time_ranges") or [])) / 3.0
    popularity = float(artist.get("popularity") or 50) / 100.0
    return (1.0 / math.sqrt(rank)) * 45.0 + blend_score * 0.35 + durability * 14.0 + popularity * 4.0


def build_user_taste_clusters(
    top_artists: List[Dict[str, Any]],
    top_tracks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cluster_scores: Dict[str, float] = defaultdict(float)
    cluster_artists: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    artist_clusters: Dict[str, List[str]] = {}

    track_counts: Dict[str, float] = defaultdict(float)
    for track in top_tracks:
        name = _normalize(track.get("artist"))
        if name:
            rank = max(1, int(track.get("rank") or 50))
            track_counts[name] += max(1.0, 30.0 - rank * 0.45)

    for artist in top_artists:
        name = artist.get("artist")
        if not name:
            continue
        genre_text = " ".join(artist.get("genres") or [])
        matches = match_cluster_scores(genre_text)
        base_weight = artist_weight(artist)
        artist_norm = _normalize(name)
        if track_counts.get(artist_norm):
            base_weight += min(track_counts[artist_norm], 25.0)

        if not matches:
            artist_clusters[artist_norm] = []
            continue

        total_match = sum(matches.values()) or 1.0
        ordered = sorted(matches.items(), key=lambda item: item[1], reverse=True)
        artist_clusters[artist_norm] = [key for key, _ in ordered]
        for key, raw_score in ordered:
            weighted = base_weight * (raw_score / total_match)
            cluster_scores[key] += weighted
            cluster_artists[key].append({
                "artist": name,
                "rank": artist.get("rank"),
                "weight": weighted,
                "genres": artist.get("genres") or [],
                "time_ranges": artist.get("time_ranges") or [],
            })

    max_score = max(cluster_scores.values(), default=0.0)
    normalized_scores = {
        key: round((value / max_score) * 100.0, 2) if max_score else 0.0
        for key, value in cluster_scores.items()
    }
    for key in cluster_artists:
        cluster_artists[key] = sorted(cluster_artists[key], key=lambda item: item["weight"], reverse=True)

    dominant = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "scores": normalized_scores,
        "artists": dict(cluster_artists),
        "artist_clusters": artist_clusters,
        "dominant_clusters": dominant,
    }


def classify_event_cluster(event: Dict[str, Any], direct_matches: List[str] | None = None, taste_clusters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    source_text = " ".join([
        str(event.get("genre") or ""),
        str(event.get("subgenre") or ""),
        str(event.get("segment") or ""),
        str(event.get("event_name") or ""),
    ])
    scores = match_cluster_scores(source_text)

    # If an event is a direct Spotify artist match and that artist has a known cluster,
    # use that artist metadata as a strong, trustworthy signal.
    taste_clusters = taste_clusters or {}
    artist_cluster_map = taste_clusters.get("artist_clusters") or {}
    for match in direct_matches or []:
        for cluster in artist_cluster_map.get(_normalize(match), []):
            scores[cluster] = scores.get(cluster, 0.0) + 6.0

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winning_key = ordered[0][0] if ordered else None
    top_score = ordered[0][1] if ordered else 0.0
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = 0.0
    if top_score > 0:
        confidence = min(100.0, 55.0 + top_score * 7.0 + max(0.0, top_score - second_score) * 4.0)

    return {
        "winning_genre_cluster": winning_key,
        "winning_genre_cluster_label": cluster_label(winning_key),
        "event_cluster_confidence": round(confidence, 2),
        "event_cluster_scores": scores,
    }


def anchors_for_cluster(
    taste_clusters: Dict[str, Any],
    cluster_key: str | None,
    direct_matches: List[str] | None = None,
    limit: int = 3,
) -> List[str]:
    if direct_matches:
        return [name for name in direct_matches if name][:limit]
    if not cluster_key:
        return []
    candidates = (taste_clusters.get("artists") or {}).get(cluster_key, [])
    return [item.get("artist") for item in candidates if item.get("artist")][:limit]


def event_cluster_feature_values(cluster_key: str | None) -> Dict[str, float]:
    return {f"cluster_{key}": 1.0 if key == cluster_key else 0.0 for key in CLUSTER_KEYS}
