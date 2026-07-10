from typing import List, Dict, Any, Optional
import requests
from .config import get_secret

SEARCH_LOCATIONS_URL = "https://api.songkick.com/api/3.0/search/locations.json"
METRO_EVENTS_URL = "https://api.songkick.com/api/3.0/metro_areas/{metro_id}/calendar.json"

def _parse_songkick_event(event: Dict[str, Any]) -> Dict[str, Any]:
    venue = event.get("venue") or {}
    location = event.get("location") or {}
    performances = event.get("performance") or []
    start = event.get("start") or {}
    artists = []
    for perf in performances:
        artist = perf.get("artist") or {}
        if artist.get("displayName"):
            artists.append(artist["displayName"])
    return {
        "source": "Songkick",
        "source_event_id": event.get("id"),
        "event_id": f"sk_{event.get('id')}",
        "event_name": event.get("displayName"),
        "date": start.get("date"),
        "time": start.get("time"),
        "url": event.get("uri"),
        "status": event.get("status"),
        "venue": venue.get("displayName"),
        "city": (location.get("city") or "").split(",")[0] if location.get("city") else None,
        "state": None,
        "country": None,
        "latitude": venue.get("lat"),
        "longitude": venue.get("lng"),
        "artists": artists,
        "segment": "Music",
        "genre": "Music",
        "subgenre": event.get("type"),
        "min_price": None,
        "max_price": None,
        "price_source": None,
        "image_url": None,
    }

def _find_metro_area(city: str, state_code: str = "TX") -> Optional[int]:
    api_key = get_secret("SONGKICK_API_KEY")
    if not api_key:
        return None
    params = {"apikey": api_key, "query": f"{city}, {state_code}"}
    response = requests.get(SEARCH_LOCATIONS_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    locations = payload.get("resultsPage", {}).get("results", {}).get("location", [])
    if not locations:
        return None
    return (locations[0].get("metroArea") or {}).get("id")

def search_music_events(
    city: str = "Austin",
    state_code: str = "TX",
    country_code: str = "US",
    radius: int = 50,
    size: int = 100,
    keyword: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    venue_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    api_key = get_secret("SONGKICK_API_KEY")
    if not api_key:
        return []
    metro_id = _find_metro_area(city, state_code)
    if not metro_id:
        return []
    params = {"apikey": api_key, "per_page": min(size, 50), "page": 1}
    if start_date:
        params["min_date"] = start_date
    if end_date:
        params["max_date"] = end_date
    response = requests.get(METRO_EVENTS_URL.format(metro_id=metro_id), params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    events = [_parse_songkick_event(e) for e in payload.get("resultsPage", {}).get("results", {}).get("event", [])]
    if keyword:
        kw = keyword.lower()
        events = [e for e in events if kw in str(e.get("event_name", "")).lower() or kw in " ".join(e.get("artists", []) or []).lower() or kw in str(e.get("venue", "")).lower()]
    return events[:size]
