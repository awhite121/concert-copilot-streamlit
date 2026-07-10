from typing import List, Dict, Any
import re
from collections import defaultdict

def _clean_name(name: str) -> str:
    if not name:
        return ""
    value = name.lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def _collapse_date_variants_name(name: str) -> str:
    """
    Keeps ACL Weekend One and Weekend Two separate,
    but removes day/date variants like "- Day 1", "(Friday)", etc.
    """
    value = _clean_name(name)
    value = re.sub(r"\bday\s*\d+\b", "", value)
    value = re.sub(r"\bfriday\b|\bsaturday\b|\bsunday\b|\bthursday\b", "", value)
    value = re.sub(r"\b\d{1,2}/\d{1,2}\b", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def _collapse_series_name(name: str) -> str:
    """
    Collapses broader festival/tour series.
    Example: Austin City Limits Music Festival - Weekend One/Two -> Austin City Limits Music Festival.
    """
    value = _collapse_date_variants_name(name)
    value = re.sub(r"\bweekend\s+(one|two|three|1|2|3)\b", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def group_key(event: Dict[str, Any], mode: str):
    venue = _clean_name(event.get("venue") or "")
    city = _clean_name(event.get("city") or "")
    name = event.get("event_name") or ""

    if mode == "Show all dates":
        return None

    if mode == "Collapse festival series":
        clean = _collapse_series_name(name)
    else:
        clean = _collapse_date_variants_name(name)

    # Only group when the name is long enough to avoid merging unrelated short artist names.
    if len(clean) < 8:
        return None

    return (clean, venue, city)

def collapse_ranked_events(events: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "Show all dates":
        return events

    grouped = defaultdict(list)
    passthrough = []

    for event in events:
        key = group_key(event, mode)
        if key is None:
            passthrough.append(event)
        else:
            grouped[key].append(event)

    collapsed = []
    for _, group in grouped.items():
        # Ranked list is already sorted, so first is best.
        primary = dict(group[0])
        dates = sorted({str(e.get("date")) for e in group if e.get("date")})
        times = sorted({str(e.get("time")) for e in group if e.get("time")})
        urls = []
        for e in group:
            for u in (e.get("all_urls") or [e.get("url")]):
                if u and u not in urls:
                    urls.append(u)

        primary["grouped_event_count"] = len(group)
        primary["available_dates"] = dates
        primary["available_times"] = times
        primary["all_urls"] = urls or primary.get("all_urls")
        if len(group) > 1:
            if len(dates) > 1:
                primary["grouping_note"] = f"{len(group)} related listings collapsed · dates: {', '.join(dates[:4])}" + ("..." if len(dates) > 4 else "")
            else:
                primary["grouping_note"] = f"{len(group)} related listings collapsed"
        else:
            primary["grouping_note"] = None
        collapsed.append(primary)

    combined = collapsed + passthrough
    return sorted(combined, key=lambda x: x.get("final_score", 0), reverse=True)
