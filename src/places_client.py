from typing import List, Dict, Any
import requests
from math import radians, sin, cos, sqrt, atan2
from .config import get_secret

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
YELP_SEARCH_URL = "https://api.yelp.com/v3/businesses/search"

def haversine_miles(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except Exception:
        return None
    R = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def places_configured() -> bool:
    return bool(get_secret("GOOGLE_MAPS_API_KEY"))

def yelp_configured() -> bool:
    return bool(get_secret("YELP_API_KEY"))

def _norm(v):
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum() or ch.isspace()).strip()

def search_google_places(venue_name, city, state, latitude=None, longitude=None, place_kind="restaurants bars coffee", max_results=8, radius_miles=0.75):
    key = get_secret("GOOGLE_MAPS_API_KEY")
    if not key or not venue_name:
        return []
    radius_meters = max(100, int(float(radius_miles) * 1609.34))
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.location,places.googleMapsUri,places.primaryTypeDisplayName,places.currentOpeningHours",
    }
    body = {"textQuery": f"{place_kind} near {venue_name}, {city}, {state}", "maxResultCount": max_results}
    if latitude and longitude:
        body["locationBias"] = {"circle": {"center": {"latitude": float(latitude), "longitude": float(longitude)}, "radius": float(radius_meters)}}
    try:
        r = requests.post(TEXT_SEARCH_URL, headers=headers, json=body, timeout=25)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return []
    out = []
    for p in payload.get("places", []):
        loc = p.get("location") or {}
        miles = haversine_miles(latitude, longitude, loc.get("latitude"), loc.get("longitude")) if latitude and longitude else None
        if miles is not None and miles > radius_miles:
            continue
        out.append({
            "source": "Google Places",
            "name": (p.get("displayName") or {}).get("text"),
            "type": (p.get("primaryTypeDisplayName") or {}).get("text"),
            "address": p.get("formattedAddress"),
            "rating": p.get("rating"),
            "price_level": p.get("priceLevel"),
            "maps_url": p.get("googleMapsUri"),
            "url": p.get("googleMapsUri"),
            "open_now": (p.get("currentOpeningHours") or {}).get("openNow"),
            "distance_miles": round(miles, 2) if miles is not None else None,
        })
    return sorted(out, key=lambda x: (999 if x.get("distance_miles") is None else x.get("distance_miles"), -(x.get("rating") or 0)))

def search_yelp_places(venue_name, city, state, latitude=None, longitude=None, place_kind="restaurants,bars,coffee", max_results=8, radius_miles=0.75):
    key = get_secret("YELP_API_KEY")
    if not key:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    params = {"term": "restaurants bars coffee", "categories": place_kind, "limit": min(max_results, 20), "radius": max(100, min(40000, int(float(radius_miles)*1609.34))), "sort_by": "best_match"}
    if latitude and longitude:
        params["latitude"] = float(latitude)
        params["longitude"] = float(longitude)
    else:
        params["location"] = f"{venue_name}, {city}, {state}"
    try:
        r = requests.get(YELP_SEARCH_URL, headers=headers, params=params, timeout=25)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return []
    out=[]
    for b in payload.get("businesses", []):
        coords = b.get("coordinates") or {}
        miles = haversine_miles(latitude, longitude, coords.get("latitude"), coords.get("longitude")) if latitude and longitude else None
        if miles is not None and miles > radius_miles:
            continue
        loc = b.get("location") or {}
        cats = b.get("categories") or []
        out.append({
            "source": "Yelp",
            "name": b.get("name"),
            "type": ", ".join([c.get("title") for c in cats if c.get("title")]) or None,
            "address": ", ".join([x for x in (loc.get("display_address") or []) if x]),
            "rating": b.get("rating"),
            "price_level": b.get("price"),
            "maps_url": None,
            "url": b.get("url"),
            "open_now": not b.get("is_closed") if b.get("is_closed") is not None else None,
            "distance_miles": round(miles, 2) if miles is not None else (round((b.get("distance") or 0)/1609.34, 2) if b.get("distance") else None),
        })
    return sorted(out, key=lambda x: (999 if x.get("distance_miles") is None else x.get("distance_miles"), -(x.get("rating") or 0)))

STYLE_QUERIES = {
    "casual": ["tacos burgers brewery patio casual restaurant", "beer garden casual bar", "coffee dessert"],
    "date night": ["cocktail bar romantic restaurant wine bar", "dinner reservations", "dessert wine"],
    "high energy": ["live music bar dance bar late night food", "cocktail bar", "sports bar"],
    "chill": ["quiet wine bar coffee relaxed patio", "dessert coffee", "casual restaurant"],
    "foodie": ["best restaurant chef driven dinner", "wine bar", "dessert"],
    "pregame": ["brewery sports bar tacos", "cocktail bar", "late night food"],
    "group hang": ["brewery beer garden casual restaurant", "pizza tacos", "bar with patio"],
    "late-night after": ["late night bar cocktail lounge wine bar", "late night food after show", "nightlife bar music lounge"],
    "coffee/dessert": ["coffee dessert bakery", "ice cream dessert", "late coffee"],
}


def _style_terms(night_style: str) -> List[str]:
    style = str(night_style or "date night").lower().replace("-", " ")
    if "high" in style:
        style = "high energy"
    if "date" in style:
        style = "date night"
    if "food" in style:
        style = "foodie"
    if "pre" in style:
        style = "pregame"
    if "group" in style:
        style = "group hang"
    if "late" in style or "after" in style:
        style = "late-night after"
    if "coffee" in style or "dessert" in style:
        style = "coffee/dessert"
    if style not in STYLE_QUERIES:
        style = "date night"
    return STYLE_QUERIES[style]


def _place_score(place, style_terms):
    text = f"{place.get('name','')} {place.get('type','')}".lower()
    category_hit = any(term in text for query in style_terms for term in query.split() if len(term) > 4)
    rating = float(place.get("rating") or 0.0)
    distance = place.get("distance_miles")
    distance_score = max(0.0, 1.5 - float(distance or 1.5)) * 2.0
    return rating * 2.0 + distance_score + (3.0 if category_hit else 0.0)


def search_nearby_plan_places(venue_name, city, state, latitude=None, longitude=None, place_kind="restaurants bars coffee", max_results=10, radius_miles=0.75, use_google=True, use_yelp=True, night_style="date night"):
    style_queries = _style_terms(night_style)
    results=[]
    for query in style_queries:
        if use_google:
            results.extend(search_google_places(venue_name, city, state, latitude, longitude, query, max(4, max_results//2), radius_miles))
        if use_yelp:
            # Yelp category search is stricter; use term for style and broad categories for recall.
            results.extend(search_yelp_places(venue_name, city, state, latitude, longitude, "restaurants,bars,coffee,nightlife", max(4, max_results//2), radius_miles))
    deduped=[]; seen=set()
    for p in results:
        key=(_norm(p.get("name")), _norm(p.get("address"))[:32])
        if key in seen: continue
        seen.add(key)
        p = dict(p)
        p["night_style_score"] = round(_place_score(p, style_queries), 2)
        deduped.append(p)
    return sorted(deduped, key=lambda x: (-float(x.get("night_style_score") or 0), 999 if x.get("distance_miles") is None else x.get("distance_miles")))[:max_results]

def test_google_places_key(radius_miles=0.75) -> Dict[str, Any]:
    key = get_secret("GOOGLE_MAPS_API_KEY")
    if not key:
        return {"ok": False, "message": "GOOGLE_MAPS_API_KEY is not set."}
    places = search_google_places("Moody Center ATX", "Austin", "TX", 30.2810, -97.7320, max_results=3, radius_miles=radius_miles)
    return {"ok": bool(places), "message": "Google Places returned nearby results." if places else "No Google places returned. Check key, billing, Places API (New), and radius.", "sample_count": len(places), "sample": places[:3]}

def test_yelp_key(radius_miles=0.75) -> Dict[str, Any]:
    key = get_secret("YELP_API_KEY")
    if not key:
        return {"ok": False, "message": "YELP_API_KEY is not set."}
    places = search_yelp_places("Moody Center ATX", "Austin", "TX", 30.2810, -97.7320, max_results=3, radius_miles=radius_miles)
    return {"ok": bool(places), "message": "Yelp returned nearby results." if places else "No Yelp places returned. Check key/access/radius.", "sample_count": len(places), "sample": places[:3]}
