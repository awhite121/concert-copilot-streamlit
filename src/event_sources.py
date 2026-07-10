from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ticketmaster_client import search_music_events as search_ticketmaster
from .seatgeek_client import search_music_events as search_seatgeek, search_price_for_event
from .songkick_client import search_music_events as search_songkick


def normalize_name(value: str) -> str:
    if not value:
        return ""
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _event_identity(event: Dict[str, Any]) -> str:
    artists = sorted({normalize_name(a) for a in (event.get("artists") or []) if a})
    if artists:
        return " ".join(artists[:3])
    return normalize_name(event.get("event_name"))


def dedupe_key(event: Dict[str, Any]) -> Tuple[str, str, str]:
    # City is more stable than venue wording across ticketing sources.
    return (
        _event_identity(event)[:100],
        str(event.get("date") or ""),
        normalize_name(event.get("city") or event.get("venue"))[:80],
    )


def _similar_event(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if str(a.get("date") or "") != str(b.get("date") or ""):
        return False
    if normalize_name(a.get("city")) and normalize_name(b.get("city")):
        if normalize_name(a.get("city")) != normalize_name(b.get("city")):
            return False
    title_score = SequenceMatcher(None, normalize_name(a.get("event_name")), normalize_name(b.get("event_name"))).ratio()
    artist_score = SequenceMatcher(None, _event_identity(a), _event_identity(b)).ratio()
    venue_score = SequenceMatcher(None, normalize_name(a.get("venue")), normalize_name(b.get("venue"))).ratio()
    return artist_score >= 0.82 or (title_score >= 0.70 and venue_score >= 0.45)


def _date_in_range(event: Dict[str, Any], start_date: Optional[str], end_date: Optional[str]) -> bool:
    event_date = event.get("date")
    if not event_date:
        return True
    try:
        value = datetime.strptime(event_date[:10], "%Y-%m-%d").date()
        if start_date and value < datetime.strptime(start_date, "%Y-%m-%d").date():
            return False
        if end_date and value > datetime.strptime(end_date, "%Y-%m-%d").date():
            return False
        return True
    except Exception:
        return True


def _venue_matches(event: Dict[str, Any], venue_name: Optional[str]) -> bool:
    if not venue_name:
        return True
    return normalize_name(venue_name) in normalize_name(event.get("venue", ""))


def _post_filter(events: List[Dict[str, Any]], start_date=None, end_date=None, venue_name=None) -> List[Dict[str, Any]]:
    return [event for event in events if _date_in_range(event, start_date, end_date) and _venue_matches(event, venue_name)]


def _valid_prices(group, field):
    values = []
    for event in group:
        value = event.get(field)
        if isinstance(value, (int, float)) and value > 0:
            values.append(float(value))
    return values


def _merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    def richness(event):
        return sum([
            event.get("min_price") is not None,
            event.get("median_price") is not None,
            event.get("image_url") is not None,
            event.get("url") is not None,
            event.get("latitude") is not None,
            len(event.get("artists") or []) > 0,
        ])

    primary = sorted(group, key=richness, reverse=True)[0].copy()
    source_values = set()
    retrieval_values = set()
    urls = []
    for event in group:
        for source in (event.get("sources") or [event.get("source", "Unknown")]):
            if source:
                source_values.add(source)
        for method in (event.get("retrieval_methods") or [event.get("retrieval_method", "broad_city")]):
            if method:
                retrieval_values.add(method)
        for url in (event.get("all_urls") or [event.get("url")]):
            if url and url not in urls:
                urls.append(url)
    sources = sorted(source_values)
    retrieval_methods = sorted(retrieval_values)

    min_prices = _valid_prices(group, "min_price")
    max_prices = _valid_prices(group, "max_price")
    median_prices = _valid_prices(group, "median_price")
    average_prices = _valid_prices(group, "average_price")

    if min_prices:
        primary["min_price"] = min(min_prices)
        price_event = next((event for event in group if event.get("min_price") == primary["min_price"]), None)
        primary["price_source"] = (price_event or {}).get("price_source")
        primary["price_type"] = (price_event or {}).get("price_type") or "starting"
    if max_prices:
        primary["max_price"] = max(max_prices)
    if median_prices:
        primary["median_price"] = min(median_prices)
    if average_prices:
        primary["average_price"] = min(average_prices)

    for event in group:
        for field in [
            "image_url", "latitude", "longitude", "genre", "subgenre", "time",
            "listing_count", "all_inclusive_pricing", "sale_start", "sale_end",
        ]:
            if primary.get(field) is None and event.get(field) is not None:
                primary[field] = event[field]

    primary["sources"] = sources
    primary["source"] = " + ".join(sources)
    primary["source_count"] = len(sources)
    primary["retrieval_methods"] = retrieval_methods
    primary["all_urls"] = urls
    primary["dedupe_count"] = sum(int(event.get("dedupe_count") or 1) for event in group)
    return primary


def merge_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exact_groups = defaultdict(list)
    for event in events:
        exact_groups[dedupe_key(event)].append(event)

    initial = [_merge_group(group) for group in exact_groups.values()]

    # Second conservative fuzzy pass catches title differences such as
    # "Artist - Tour" versus "Artist with Support" across ticketing sources.
    clusters: List[List[Dict[str, Any]]] = []
    for event in initial:
        placed = False
        for cluster in clusters:
            if _similar_event(event, cluster[0]):
                cluster.append(event)
                placed = True
                break
        if not placed:
            clusters.append([event])

    merged = [_merge_group(cluster) for cluster in clusters]
    return sorted(merged, key=lambda event: (str(event.get("date") or "9999-99-99"), str(event.get("time") or "")))


def _unique_names(names: List[str], max_names: int = 25) -> List[str]:
    output, seen = [], set()
    for name in names or []:
        if not name:
            continue
        name = str(name).strip()
        key = normalize_name(name)
        if key and key not in seen:
            seen.add(key)
            output.append(name)
        if len(output) >= max_names:
            break
    return output


def _enrich_prices(events, city, state_code, targeted_artist_names, limit=24):
    """Enrich the most relevant missing-price events concurrently.

    The broad source pull and dedupe happen first. Then this checks SeatGeek for
    missing prices using conservative date/title/artist/venue matching. Concurrent
    calls keep the UI responsive enough for a portfolio app.
    """
    target_set = {normalize_name(name) for name in targeted_artist_names or []}

    def priority(event):
        artists = {normalize_name(name) for name in (event.get("artists") or [])}
        direct_target = bool(artists & target_set)
        multi_source = int(event.get("source_count") or 1) > 1
        return (
            0 if direct_target else 1,
            0 if multi_source else 1,
            str(event.get("date") or "9999-99-99"),
        )

    missing = [
        event for event in events
        if event.get("min_price") is None
        and event.get("median_price") is None
        and event.get("average_price") is None
    ]
    selected = sorted(missing, key=priority)[:max(0, int(limit))]
    if not selected:
        return events, 0

    def fetch(event):
        return event, search_price_for_event(event, city, state_code)

    enriched_count = 0
    with ThreadPoolExecutor(max_workers=min(5, len(selected))) as executor:
        futures = [executor.submit(fetch, event) for event in selected]
        for future in as_completed(futures):
            try:
                event, match = future.result()
            except Exception:
                continue
            if not match:
                continue
            for field in [
                "min_price", "max_price", "median_price", "average_price",
                "listing_count", "price_type", "price_source",
            ]:
                if match.get(field) is not None:
                    event[field] = match[field]
            if match.get("url"):
                urls = list(event.get("all_urls") or [event.get("url")])
                if match["url"] not in urls:
                    urls.append(match["url"])
                event["all_urls"] = [url for url in urls if url]
            if match.get("image_url") and not event.get("image_url"):
                event["image_url"] = match["image_url"]
            sources = set(event.get("sources") or [event.get("source")])
            sources.add("SeatGeek")
            event["sources"] = sorted(source for source in sources if source)
            event["source"] = " + ".join(event["sources"])
            event["source_count"] = len(event["sources"])
            event["price_enriched"] = True
            enriched_count += 1
    return events, enriched_count

def search_all_sources(
    city: str = "Austin",
    state_code: str = "TX",
    country_code: str = "US",
    radius: int = 50,
    size: int = 100,
    keyword: Optional[str] = None,
    use_ticketmaster: bool = True,
    use_seatgeek: bool = True,
    use_songkick: bool = False,
    ticketmaster_pages: int = 5,
    seatgeek_pages: int = 2,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    venue_name: Optional[str] = None,
    targeted_artist_names: Optional[List[str]] = None,
    max_targeted_artists: int = 20,
    smart_artist_search: bool = True,
    price_enrichment_limit: int = 40,
) -> Dict[str, Any]:
    raw, errors, coverage_rows = [], {}, []
    targeted_artist_names = _unique_names(targeted_artist_names or [], max_targeted_artists)

    def record(source, method, count, artist=None, error=None):
        coverage_rows.append({"source": source, "method": method, "artist": artist, "events_found": count, "error": error})

    if use_ticketmaster:
        try:
            batch = search_ticketmaster(city, state_code, country_code, radius, min(size, 200), keyword, ticketmaster_pages, start_date, end_date, venue_name, retrieval_method="broad_city")
            raw.extend(batch)
            record("Ticketmaster", "broad_city", len(batch))
        except Exception as exc:
            errors["Ticketmaster"] = str(exc)
            record("Ticketmaster", "broad_city", 0, error=str(exc))

    if use_seatgeek:
        try:
            batch = search_seatgeek(city, state_code, country_code, radius, size, keyword, start_date, end_date, venue_name, pages=seatgeek_pages, retrieval_method="broad_city")
            raw.extend(batch)
            record("SeatGeek", "broad_city", len(batch))
        except Exception as exc:
            errors["SeatGeek"] = str(exc)
            record("SeatGeek", "broad_city", 0, error=str(exc))

    if use_songkick:
        try:
            batch = search_songkick(city, state_code, country_code, radius, size, keyword, start_date, end_date, venue_name)
            raw.extend(batch)
            record("Songkick", "broad_city", len(batch))
        except Exception as exc:
            errors["Songkick"] = str(exc)
            record("Songkick", "broad_city", 0, error=str(exc))

    if smart_artist_search and targeted_artist_names:
        for artist in targeted_artist_names:
            if use_ticketmaster:
                try:
                    batch = search_ticketmaster(city, state_code, country_code, radius, min(size, 100), artist, min(ticketmaster_pages, 2), start_date, end_date, venue_name, retrieval_method=f"artist_target:{artist}")
                    raw.extend(batch)
                    record("Ticketmaster", "artist_target", len(batch), artist=artist)
                except Exception as exc:
                    record("Ticketmaster", "artist_target", 0, artist=artist, error=str(exc))
            if use_seatgeek:
                try:
                    batch = search_seatgeek(city, state_code, country_code, radius, min(size, 100), artist, start_date, end_date, venue_name, pages=1, retrieval_method=f"artist_target:{artist}")
                    raw.extend(batch)
                    record("SeatGeek", "artist_target", len(batch), artist=artist)
                except Exception as exc:
                    record("SeatGeek", "artist_target", 0, artist=artist, error=str(exc))

    raw_filtered = _post_filter(raw, start_date=start_date, end_date=end_date, venue_name=venue_name)
    merged = merge_events(raw_filtered)
    price_enriched = 0
    if use_seatgeek and price_enrichment_limit:
        merged, price_enriched = _enrich_prices(
            merged, city, state_code, targeted_artist_names, limit=price_enrichment_limit
        )

    final_text = normalize_name(" || ".join([
        " ".join([str(event.get("event_name") or ""), " ".join(event.get("artists") or [])])
        for event in merged
    ]))
    for row in coverage_rows:
        artist = row.get("artist")
        row["in_final_pool"] = None if not artist else normalize_name(artist) in final_text

    counts = {
        "raw_total": len(raw),
        "after_filters": len(raw_filtered),
        "deduped_total": len(merged),
        "price_enriched": price_enriched,
        "by_source": {},
        "errors": errors,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "venue_name": venue_name,
            "targeted_artist_count": len(targeted_artist_names),
        },
        "coverage_rows": coverage_rows,
        "targeted_artist_names": targeted_artist_names,
    }
    for event in raw:
        source = event.get("source", "Unknown")
        counts["by_source"][source] = counts["by_source"].get(source, 0) + 1
    return {"events": merged, "counts": counts, "raw_events": raw}
