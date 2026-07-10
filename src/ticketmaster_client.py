from typing import List, Dict, Any, Optional
import requests
from .config import get_secret

BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"


def _safe_get(dct, path, default=None):
    cur = dct
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _iso_start(date_value):
    return f"{date_value}T00:00:00Z" if date_value else None


def _iso_end(date_value):
    return f"{date_value}T23:59:59Z" if date_value else None


def parse_price_ranges(event: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ranges = event.get("priceRanges") or []
    mins = [r.get("min") for r in ranges if isinstance(r.get("min"), (int, float))]
    maxs = [r.get("max") for r in ranges if isinstance(r.get("max"), (int, float))]
    return {
        "min_price": min(mins) if mins else None,
        "max_price": max(maxs) if maxs else None,
    }


def _best_image(event):
    images = event.get("images") or []
    if not images:
        return None
    return sorted(images, key=lambda image: (image.get("width", 0) * image.get("height", 0)), reverse=True)[0].get("url")


def _parse_ticketmaster_event(event: Dict[str, Any], retrieval_method: str = "broad_city") -> Dict[str, Any]:
    venue = event.get("_embedded", {}).get("venues", [{}])[0]
    attractions = event.get("_embedded", {}).get("attractions", [])
    classifications = event.get("classifications", [{}])
    classification = classifications[0] if classifications else {}
    price = parse_price_ranges(event)
    return {
        "source": "Ticketmaster",
        "retrieval_method": retrieval_method,
        "source_event_id": event.get("id"),
        "event_id": f"tm_{event.get('id')}",
        "event_name": event.get("name"),
        "date": _safe_get(event, ["dates", "start", "localDate"]),
        "time": _safe_get(event, ["dates", "start", "localTime"]),
        "url": event.get("url"),
        "status": _safe_get(event, ["dates", "status", "code"]),
        "venue": venue.get("name"),
        "city": _safe_get(venue, ["city", "name"]),
        "state": _safe_get(venue, ["state", "stateCode"]),
        "country": _safe_get(venue, ["country", "countryCode"]),
        "latitude": _safe_get(venue, ["location", "latitude"]),
        "longitude": _safe_get(venue, ["location", "longitude"]),
        "artists": [a.get("name") for a in attractions if a.get("name")],
        "segment": _safe_get(classification, ["segment", "name"]),
        "genre": _safe_get(classification, ["genre", "name"]),
        "subgenre": _safe_get(classification, ["subGenre", "name"]),
        "min_price": price["min_price"],
        "max_price": price["max_price"],
        "median_price": None,
        "average_price": None,
        "listing_count": None,
        "price_type": "range" if price["min_price"] is not None else None,
        "price_source": "Ticketmaster published range" if price["min_price"] is not None else None,
        "all_inclusive_pricing": event.get("allInclusivePricing"),
        "sale_start": _safe_get(event, ["sales", "public", "startDateTime"]),
        "sale_end": _safe_get(event, ["sales", "public", "endDateTime"]),
        "image_url": _best_image(event),
    }


def search_music_events(
    city: str = "Austin",
    state_code: str = "TX",
    country_code: str = "US",
    radius: int = 50,
    size: int = 100,
    keyword: Optional[str] = None,
    pages: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    venue_name: Optional[str] = None,
    retrieval_method: str = "broad_city",
) -> List[Dict[str, Any]]:
    api_key = get_secret("TICKETMASTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TICKETMASTER_API_KEY")

    all_events = []
    page_size = min(size, 200)
    max_pages = max(1, min(pages, 10))
    combined_keyword = keyword or (venue_name if venue_name else None)

    for page in range(max_pages):
        if page_size * page >= 1000:
            break
        params = {
            "apikey": api_key,
            "classificationName": "music",
            "countryCode": country_code,
            "city": city,
            "stateCode": state_code,
            "radius": radius,
            "unit": "miles",
            "size": page_size,
            "page": page,
            "sort": "date,asc",
        }
        if combined_keyword:
            params["keyword"] = combined_keyword
        if start_date:
            params["startDateTime"] = _iso_start(start_date)
        if end_date:
            params["endDateTime"] = _iso_end(end_date)

        response = requests.get(BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        raw_events = payload.get("_embedded", {}).get("events", [])
        if not raw_events:
            break
        all_events.extend(_parse_ticketmaster_event(event, retrieval_method=retrieval_method) for event in raw_events)
    return all_events
