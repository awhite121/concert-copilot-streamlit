from typing import List, Dict, Any, Optional
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from .config import get_secret

SCOPE = "user-top-read"

def get_spotify_client(cache_key: str = "default") -> Optional[spotipy.Spotify]:
    client_id = get_secret("SPOTIFY_CLIENT_ID")
    client_secret = get_secret("SPOTIFY_CLIENT_SECRET")
    redirect_uri = get_secret("SPOTIFY_REDIRECT_URI", "http://localhost:8501")

    if not client_id or not client_secret:
        return None

    safe_key = "".join(ch for ch in cache_key if ch.isalnum() or ch in ["_", "-"])[:80]
    cache_path = f".spotify_cache_{safe_key}"

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPE,
        cache_path=cache_path,
        show_dialog=False,
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def get_current_user(sp: spotipy.Spotify) -> Dict[str, Any]:
    user = sp.current_user()
    return {
        "user_id": user.get("id") or user.get("display_name") or "spotify_user",
        "display_name": user.get("display_name") or user.get("id") or "Spotify User",
        "spotify_url": user.get("external_urls", {}).get("spotify"),
    }

def get_top_artists(sp: spotipy.Spotify, limit: int = 30, time_range: str = "medium_term") -> List[Dict[str, Any]]:
    results = sp.current_user_top_artists(limit=limit, time_range=time_range)
    artists = []
    for rank, item in enumerate(results.get("items", []), start=1):
        artists.append({
            "rank": rank,
            "artist_id": item.get("id"),
            "artist": item.get("name"),
            "genres": item.get("genres", []) or [],
            "popularity": item.get("popularity"),
            "spotify_url": item.get("external_urls", {}).get("spotify"),
            "image_url": item.get("images", [{}])[0].get("url") if item.get("images") else None,
        })
    return artists

def get_top_tracks(sp: spotipy.Spotify, limit: int = 30, time_range: str = "medium_term") -> List[Dict[str, Any]]:
    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    tracks = []
    for rank, item in enumerate(results.get("items", []), start=1):
        artists = item.get("artists", [])
        tracks.append({
            "rank": rank,
            "track_id": item.get("id"),
            "track": item.get("name"),
            "artist": artists[0].get("name") if artists else None,
            "artist_ids": [a.get("id") for a in artists if a.get("id")],
            "popularity": item.get("popularity"),
            "spotify_url": item.get("external_urls", {}).get("spotify"),
        })
    return tracks

def demo_profile():
    """
    Lets recruiters test the app without logging into Spotify.
    Swap this to match your actual music taste before publishing.
    """
    user = {
        "user_id": "demo_user",
        "display_name": "Demo User",
        "spotify_url": None,
    }

    top_artists = [
        {"rank": 1, "artist": "Fred again..", "genres": ["edm", "house", "electronic"], "popularity": 88, "spotify_url": None},
        {"rank": 2, "artist": "Zach Bryan", "genres": ["country", "singer-songwriter", "folk"], "popularity": 91, "spotify_url": None},
        {"rank": 3, "artist": "Noah Kahan", "genres": ["folk-pop", "singer-songwriter"], "popularity": 87, "spotify_url": None},
        {"rank": 4, "artist": "ODESZA", "genres": ["electronic", "indietronica"], "popularity": 77, "spotify_url": None},
        {"rank": 5, "artist": "Mt. Joy", "genres": ["indie rock", "folk rock"], "popularity": 72, "spotify_url": None},
        {"rank": 6, "artist": "Tyler Childers", "genres": ["country", "bluegrass", "americana"], "popularity": 82, "spotify_url": None},
        {"rank": 7, "artist": "SZA", "genres": ["r&b", "pop"], "popularity": 91, "spotify_url": None},
        {"rank": 8, "artist": "Tame Impala", "genres": ["psychedelic pop", "indie"], "popularity": 85, "spotify_url": None},
    ]
    top_tracks = [
        {"rank": 1, "track": "Delilah", "artist": "Fred again..", "popularity": 80, "spotify_url": None},
        {"rank": 2, "track": "Something in the Orange", "artist": "Zach Bryan", "popularity": 86, "spotify_url": None},
        {"rank": 3, "track": "Stick Season", "artist": "Noah Kahan", "popularity": 88, "spotify_url": None},
        {"rank": 4, "track": "Sun Models", "artist": "ODESZA", "popularity": 70, "spotify_url": None},
        {"rank": 5, "track": "Silver Lining", "artist": "Mt. Joy", "popularity": 70, "spotify_url": None},
    ]
    return user, top_artists, top_tracks


def get_blended_taste_profile(sp: spotipy.Spotify, artist_limit: int = 40, track_limit: int = 40):
    """
    Blend short, medium, and long-term Spotify taste into one profile.
    Recent taste gets more weight, but durable favorites still matter.
    """
    range_weights = {"short_term": 1.35, "medium_term": 1.0, "long_term": 0.75}
    artist_bucket = {}
    track_bucket = {}

    for tr, weight in range_weights.items():
        for item in get_top_artists(sp, limit=artist_limit, time_range=tr):
            name = item.get("artist")
            if not name:
                continue
            key = name.lower()
            rank = int(item.get("rank") or artist_limit)
            score = weight * max(1, artist_limit + 1 - rank)
            if key not in artist_bucket:
                artist_bucket[key] = {**item, "blend_score": 0.0, "time_ranges": [], "rank_source": {}}
            artist_bucket[key]["blend_score"] += score
            artist_bucket[key]["time_ranges"].append(tr)
            artist_bucket[key]["rank_source"][tr] = rank
            if not artist_bucket[key].get("genres") and item.get("genres"):
                artist_bucket[key]["genres"] = item.get("genres")
            if not artist_bucket[key].get("image_url") and item.get("image_url"):
                artist_bucket[key]["image_url"] = item.get("image_url")

        for item in get_top_tracks(sp, limit=track_limit, time_range=tr):
            track = item.get("track")
            artist = item.get("artist")
            if not track or not artist:
                continue
            key = f"{track.lower()}::{artist.lower()}"
            rank = int(item.get("rank") or track_limit)
            score = weight * max(1, track_limit + 1 - rank)
            if key not in track_bucket:
                track_bucket[key] = {**item, "blend_score": 0.0, "time_ranges": [], "rank_source": {}}
            track_bucket[key]["blend_score"] += score
            track_bucket[key]["time_ranges"].append(tr)
            track_bucket[key]["rank_source"][tr] = rank

    artists = sorted(artist_bucket.values(), key=lambda x: x.get("blend_score", 0), reverse=True)
    tracks = sorted(track_bucket.values(), key=lambda x: x.get("blend_score", 0), reverse=True)
    for i, item in enumerate(artists, 1):
        item["rank"] = i
        item["blend_score"] = round(float(item.get("blend_score", 0)), 2)
        item["time_ranges"] = sorted(set(item.get("time_ranges", [])))
    for i, item in enumerate(tracks, 1):
        item["rank"] = i
        item["blend_score"] = round(float(item.get("blend_score", 0)), 2)
        item["time_ranges"] = sorted(set(item.get("time_ranges", [])))
    return artists[:artist_limit], tracks[:track_limit]


def add_group_listener_artists(top_artists, listener_name: str, artist_names):
    """Blend a second listener's favorite artists into the candidate taste profile."""
    if not artist_names:
        return top_artists
    blended = [dict(a) for a in top_artists]
    existing = {(a.get("artist") or "").lower() for a in blended}
    for idx, name in enumerate(artist_names, start=1):
        clean = str(name).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in existing:
            for item in blended:
                if (item.get("artist") or "").lower() == key:
                    listeners = item.setdefault("group_listeners", [])
                    if listener_name not in listeners:
                        listeners.append(listener_name)
            continue
        blended.append({
            "rank": min(idx, 20),
            "artist": clean,
            "genres": [],
            "popularity": None,
            "spotify_url": None,
            "image_url": None,
            "group_listener": listener_name,
            "group_listeners": [listener_name],
            "blend_score": max(1, 30 - idx),
        })
        existing.add(key)
    return blended
