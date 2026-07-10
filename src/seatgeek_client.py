from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from datetime import datetime, timedelta
import re
import requests
from .config import get_secret

BASE_URL = "https://api.seatgeek.com/2/events"


def _number(value):
    return float(value) if isinstance(value, (int, float)) else None


def _normalize(value: str) -> str:
    value = str(value or "").lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _parse_seatgeek_event(event: Dict[str, Any], retrieval_method: str = "broad_city") -> Dict[str, Any]:
    venue = event.get("venue") or {}
    performers = event.get("performers") or []
    stats = event.get("stats") or {}

    # Only call a real lowest listing a starting price. Average and median are
    # useful fallbacks, but should never be displayed as "From" prices.
    lowest_price = _number(stats.get("lowest_price"))
    lowest_good_deal = _number(stats.get("lowest_price_good_deals"))
    min_price = lowest_price if lowest_price is not None else lowest_good_deal
    max_price = _number(stats.get("highest_price"))
    median_price = _number(stats.get("median_price"))
    average_price = _number(stats.get("average_price"))
    listing_count = stats.get("listing_count")

    taxonomies = event.get("taxonomies") or []
    genre = taxonomies[-1].get("name") if taxonomies else "Music"
    subgenre = taxonomies[-2].get("name") if len(taxonomies) >= 2 else None
    datetime_local = event.get("datetime_local") or ""

    return {
        "source": "SeatGeek",
        "retrieval_method": retrieval_method,
        "source_event_id": event.get("id"),
        "event_id": f"sg_{event.get('id')}",
        "event_name": event.get("title") or event.get("short_title"),
        "date": datetime_local[:10] or None,
        "time": datetime_local[11:19] or None,
        "url": event.get("url"),
        "status": event.get("status"),
        "venue": venue.get("name"),
        "city": venue.get("city"),
        "state": venue.get("state"),
        "country": venue.get("country"),
        "latitude": venue.get("location", {}).get("lat") or venue.get("lat"),
        "longitude": venue.get("location", {}).get("lon") or venue.get("lon"),
        "artists": [p.get("name") for p in performers if p.get("name")],
        "segment": "Music",
        "genre": genre or "Music",
        "subgenre": subgenre,
        "min_price": min_price,
        "max_price": max_price,
        "median_price": median_price,
        "average_price": average_price,
        "listing_count": listing_count,
        "price_type": "starting" if min_price is not None else ("typical" if median_price is not None or average_price is not None else None),
        "price_source": "SeatGeek live listings" if min_price is not None else ("SeatGeek typical listing price" if median_price is not None or average_price is not None else None),
        "image_url": performers[0].get("image") if performers and performers[0].get("image") else None,
    }


def _auth_params() -> Dict[str, str]:
    client_id = get_secret("SEATGEEK_CLIENT_ID")
    client_secret = get_secret("SEATGEEK_CLIENT_SECRET")
    if not client_id:
        return {}
    params = {"client_id": client_id}
    if client_secret:
        params["client_secret"] = client_secret
    return params


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
    pages: int = 1,
    retrieval_method: str = "broad_city",
) -> List[Dict[str, Any]]:
    auth = _auth_params()
    if not auth:
        return []

    events = []
    per_page = max(1, min(size, 100))
    max_pages = max(1, min(int(pages or 1), 10))

    for page in range(1, max_pages + 1):
        params = {
            **auth,
            "per_page": per_page,
            "page": page,
            "type": "concert",
            "venue.city": city,
            "venue.state": state_code,
            "sort": "datetime_local.asc",
        }
        if keyword:
            params["q"] = keyword
        if start_date:
            params["datetime_utc.gte"] = f"{start_date}T00:00:00"
        if end_date:
            try:
                utc_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            except Exception:
                utc_end = end_date
            params["datetime_utc.lte"] = f"{utc_end}T23:59:59"

        response = requests.get(BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("events", [])
        if not batch:
            break
        events.extend([_parse_seatgeek_event(event, retrieval_method=retrieval_method) for event in batch])
        total = (payload.get("meta") or {}).get("total")
        if isinstance(total, int) and page * per_page >= total:
            break
    return events


def search_price_for_event(event: Dict[str, Any], city: str, state_code: str) -> Optional[Dict[str, Any]]:
    """Find the closest SeatGeek listing for price enrichment.

    V20 tries several conservative queries (artist, event title, artist + venue)
    and then requires same date plus title/artist/venue similarity. This improves
    price coverage without attaching a random ticket listing to the wrong show.
    """
    auth = _auth_params()
    if not auth or not event.get("date"):
        return None

    artists = event.get("artists") or []
    query_candidates = []
    if artists:
        query_candidates.append(str(artists[0]))
    if event.get("event_name"):
        query_candidates.append(str(event.get("event_name")))
    if artists and event.get("venue"):
        query_candidates.append(f"{artists[0]} {event.get('venue')}")

    seen_q = set()
    queries = []
    for q in query_candidates:
        q = q.strip()
        if q and q.lower() not in seen_q:
            seen_q.add(q.lower())
            queries.append(q)
    if not queries:
        return None

    try:
        utc_end_date = (datetime.strptime(event["date"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        utc_end_date = event["date"]

    candidates = []
    for query in queries[:3]:
        params = {
            **auth,
            "q": query,
            "venue.city": city,
            "venue.state": state_code,
            "datetime_utc.gte": f"{event['date']}T00:00:00",
            "datetime_utc.lte": f"{utc_end_date}T23:59:59",
            "per_page": 35,
            "page": 1,
            "type": "concert",
        }
        try:
            response = requests.get(BASE_URL, params=params, timeout=12)
            response.raise_for_status()
            candidates.extend(response.json().get("events", []))
        except Exception:
            continue

    target_title = _normalize(event.get("event_name"))
    target_artists = _normalize(" ".join(artists))
    target_venue = _normalize(event.get("venue"))
    best = None
    best_score = 0.0
    seen_ids = set()

    for raw in candidates:
        if raw.get("id") in seen_ids:
            continue
        seen_ids.add(raw.get("id"))
        parsed = _parse_seatgeek_event(raw, retrieval_method="price_enrichment")
        if parsed.get("date") != event.get("date"):
            continue
        title_score = SequenceMatcher(None, target_title, _normalize(parsed.get("event_name"))).ratio() if target_title else 0.0
        parsed_artists = _normalize(" ".join(parsed.get("artists") or []))
        artist_score = SequenceMatcher(None, target_artists, parsed_artists).ratio() if target_artists and parsed_artists else 0.0
        venue_score = SequenceMatcher(None, target_venue, _normalize(parsed.get("venue"))).ratio() if target_venue else 0.5
        price_bonus = 0.06 if any(parsed.get(k) is not None for k in ["min_price", "median_price", "average_price"]) else 0.0
        score = 0.50 * title_score + 0.34 * artist_score + 0.16 * venue_score + price_bonus
        if score > best_score:
            best_score = score
            best = parsed

    if best is None or best_score < 0.46:
        return None
    best["price_match_score"] = round(best_score, 3)
    return best

