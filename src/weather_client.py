from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import requests

from .config import get_secret

OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"


def weather_configured() -> bool:
    return bool(get_secret("OPENWEATHER_API_KEY"))


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _summarize_weather(row: Dict[str, Any], source: str, event_date: Optional[str] = None) -> Dict[str, Any]:
    main = row.get("main") or {}
    wind = row.get("wind") or {}
    weather = (row.get("weather") or [{}])[0]
    clouds = (row.get("clouds") or {}).get("all")
    pop = row.get("pop")
    rain = (row.get("rain") or {}).get("3h")
    snow = (row.get("snow") or {}).get("3h")
    temp = _safe_float(main.get("temp"))
    feels_like = _safe_float(main.get("feels_like"))
    wind_mph = _safe_float(wind.get("speed"))
    risk_bits = []
    if isinstance(pop, (int, float)) and pop >= 0.35:
        risk_bits.append(f"{round(pop * 100)}% precipitation chance")
    if rain:
        risk_bits.append("rain possible")
    if snow:
        risk_bits.append("snow possible")
    if temp is not None and temp <= 45:
        risk_bits.append("cool weather")
    if temp is not None and temp >= 90:
        risk_bits.append("hot weather")
    if wind_mph is not None and wind_mph >= 18:
        risk_bits.append("windy")
    return {
        "available": True,
        "source": source,
        "event_date": event_date,
        "forecast_time": row.get("dt_txt"),
        "description": weather.get("description"),
        "temperature_f": round(temp) if temp is not None else None,
        "feels_like_f": round(feels_like) if feels_like is not None else None,
        "precip_probability": round(float(pop), 2) if isinstance(pop, (int, float)) else None,
        "wind_mph": round(wind_mph, 1) if wind_mph is not None else None,
        "cloud_cover_pct": clouds,
        "risk_summary": ", ".join(risk_bits) if risk_bits else "no major weather flag from returned data",
    }


def get_event_weather(event: Dict[str, Any], city: str = "", state: str = "") -> Dict[str, Any]:
    """Return a lightweight weather context for Copilot.

    Uses OpenWeather if OPENWEATHER_API_KEY is present. Forecast API covers near-term
    dates; for dates outside the forecast window we return a clear unavailable reason.
    """
    key = get_secret("OPENWEATHER_API_KEY")
    if not key:
        return {"available": False, "reason": "OPENWEATHER_API_KEY not set"}

    lat = event.get("latitude")
    lon = event.get("longitude")
    query_city = city or event.get("city") or ""
    query_state = state or event.get("state") or ""
    event_date = str(event.get("date") or "")[:10] or None

    params = {"appid": key, "units": "imperial"}
    if lat and lon:
        params.update({"lat": lat, "lon": lon})
    elif query_city:
        params.update({"q": f"{query_city},{query_state},US" if query_state else query_city})
    else:
        return {"available": False, "reason": "event has no location for weather lookup"}

    try:
        r = requests.get(OPENWEATHER_FORECAST_URL, params=params, timeout=14)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("list") or []
        if rows and event_date:
            same_day = [row for row in rows if str(row.get("dt_txt") or "").startswith(event_date)]
            if same_day:
                # Prefer evening forecast closest to typical show time.
                target_hours = [18, 21, 15, 12]
                selected = None
                for hour in target_hours:
                    selected = next((row for row in same_day if f" {hour:02d}:" in str(row.get("dt_txt") or "")), None)
                    if selected:
                        break
                selected = selected or same_day[0]
                return _summarize_weather(selected, "OpenWeather forecast", event_date)
        if rows:
            # Forecast exists but event may be beyond API window.
            first_dt = rows[0].get("dt_txt")
            last_dt = rows[-1].get("dt_txt")
            return {
                "available": False,
                "reason": f"event date is outside returned forecast window ({first_dt} to {last_dt})",
                "source": "OpenWeather forecast",
                "event_date": event_date,
            }
    except Exception as exc:
        # Fall through to current conditions.
        last_error = str(exc)
    else:
        last_error = "no forecast rows returned"

    try:
        r = requests.get(OPENWEATHER_CURRENT_URL, params=params, timeout=14)
        r.raise_for_status()
        return _summarize_weather(r.json(), "OpenWeather current", event_date)
    except Exception as exc:
        return {"available": False, "reason": f"weather lookup failed: {exc or last_error}"}


def test_openweather_key() -> Dict[str, Any]:
    key = get_secret("OPENWEATHER_API_KEY")
    if not key:
        return {"ok": False, "message": "OPENWEATHER_API_KEY is not set."}
    sample_event = {"city": "Austin", "state": "TX", "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")}
    weather = get_event_weather(sample_event, "Austin", "TX")
    return {
        "ok": bool(weather.get("available")),
        "message": "OpenWeather returned event weather." if weather.get("available") else weather.get("reason", "No weather returned."),
        "sample": weather,
    }
