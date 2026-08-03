from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape
from typing import Any, Callable, Dict, Iterable, List
import base64
import json
import re
import urllib.parse
import uuid

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.elite_product import (
    build_playlist_ics,
    compact_reason,
    event_city,
    event_date,
    event_id,
    event_score,
    event_state,
    event_status_from_action,
    event_time,
    event_title,
    event_url,
    event_venue,
    price_text,
    spotify_url,
)


STATUS_COLORS = {
    "want_to_go": "#ff5b57",
    "maybe": "#4f73ff",
    "not_for_me": "#8d95a6",
}


def _normal(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "").strip() if ch.isalnum())


def _event_datetime(event: Dict[str, Any]) -> str:
    raw_date = event_date(event)
    if not raw_date:
        return ""
    raw_time = event_time(event)
    if raw_time:
        raw_time = raw_time[:8]
        if len(raw_time) == 5:
            raw_time += ":00"
        return f"{raw_date[:10]}T{raw_time}"
    return raw_date[:10]


def _ics_data_url(event: Dict[str, Any]) -> str:
    content = build_playlist_ics([event])
    encoded = urllib.parse.quote(content)
    return f"data:text/calendar;charset=utf-8,{encoded}"


def _dedupe_items(
    playlist_df: pd.DataFrame,
    interaction_row_to_event: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    unique: Dict[tuple, Dict[str, Any]] = {}

    for row in playlist_df.to_dict(orient="records"):
        try:
            event = interaction_row_to_event(row)
        except Exception:
            event = dict(row)

        if not isinstance(event, dict):
            continue

        # The latest stored action is authoritative.
        event = dict(event)
        event["action"] = str(row.get("action") or event.get("action") or "")
        event["created_at"] = str(row.get("created_at") or event.get("created_at") or "")

        raw_date = event_date(event) or str(row.get("event_date") or row.get("date") or "")
        parsed = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed):
            continue

        key = (
            _normal(event_title(event)),
            parsed.date().isoformat(),
            str(event_time(event))[:5],
            _normal(event_city(event)),
        )

        candidate_quality = (
            1 if event_url(event) else 0,
            1 if event.get("image_url") else 0,
            event["created_at"],
        )
        previous = unique.get(key)
        if previous is None:
            unique[key] = event
            continue

        previous_quality = (
            1 if event_url(previous) else 0,
            1 if previous.get("image_url") else 0,
            str(previous.get("created_at") or ""),
        )
        if candidate_quality > previous_quality:
            unique[key] = event

    return sorted(
        unique.values(),
        key=lambda event: (
            event_date(event),
            event_time(event),
            event_title(event),
        ),
    )


def _calendar_payload(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = []
    for event in events:
        start = _event_datetime(event)
        if not start:
            continue

        action = str(event.get("action") or "")
        ticket = event_url(event)
        spotify = spotify_url(event)
        event_identifier = event_id(event)

        payload.append({
            "id": event_identifier,
            "title": event_title(event),
            "start": start,
            "backgroundColor": STATUS_COLORS.get(action, "#ff5b57"),
            "borderColor": STATUS_COLORS.get(action, "#ff5b57"),
            "textColor": "#ffffff",
            "extendedProps": {
                "status": event_status_from_action(action),
                "venue": event_venue(event),
                "city": event_city(event),
                "state": event_state(event),
                "score": round(event_score(event), 1),
                "reason": compact_reason(event, 240),
                "price": price_text(event),
                "ticketUrl": ticket,
                "spotifyUrl": spotify,
                "calendarUrl": _ics_data_url(event),
                "planId": event_identifier,
            },
        })
    return payload


def render_elite_calendar(
    playlist_df: pd.DataFrame,
    *,
    interaction_row_to_event: Callable[[Dict[str, Any]], Dict[str, Any]],
    default_city: str = "",
    key_prefix: str = "elite_calendar",
) -> None:
    if playlist_df is None or playlist_df.empty:
        st.info("Save a show as Want or Maybe and it will appear here.")
        return

    events = _dedupe_items(playlist_df, interaction_row_to_event)
    if not events:
        st.info("Saved shows with confirmed event dates will appear here.")
        return

    city_values = sorted({event_city(event) for event in events if event_city(event)})
    default_city = str(default_city or "").strip()
    city_index = 0
    city_options = ["All cities"] + city_values
    if default_city in city_options:
        city_index = city_options.index(default_city)

    filter_a, filter_b, export_col = st.columns([1.15, 1.15, 0.8], vertical_alignment="bottom")
    with filter_a:
        status_filter = st.selectbox(
            "Shows",
            ["Want + Maybe", "Want", "Maybe", "All saved", "Don't go"],
            index=0,
            key=f"{key_prefix}_status",
        )
    with filter_b:
        city_filter = st.selectbox(
            "City",
            city_options,
            index=city_index,
            key=f"{key_prefix}_city",
        )

    filtered = list(events)
    if status_filter == "Want + Maybe":
        filtered = [event for event in filtered if event.get("action") in {"want_to_go", "maybe"}]
    elif status_filter == "Want":
        filtered = [event for event in filtered if event.get("action") == "want_to_go"]
    elif status_filter == "Maybe":
        filtered = [event for event in filtered if event.get("action") == "maybe"]
    elif status_filter == "Don't go":
        filtered = [event for event in filtered if event.get("action") == "not_for_me"]

    if city_filter != "All cities":
        filtered = [event for event in filtered if event_city(event) == city_filter]

    upcoming = []
    for event in filtered:
        parsed = pd.to_datetime(event_date(event), errors="coerce")
        if not pd.isna(parsed) and parsed.date() >= date.today():
            upcoming.append(event)

    with export_col:
        st.download_button(
            "Download calendar",
            data=build_playlist_ics(upcoming or filtered),
            file_name="encore-ai-my-shows.ics",
            mime="text/calendar",
            use_container_width=True,
            key=f"{key_prefix}_download",
            disabled=not bool(filtered),
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Showing", len(filtered))
    metric_cols[1].metric("Want", sum(event.get("action") == "want_to_go" for event in filtered))
    metric_cols[2].metric("Maybe", sum(event.get("action") == "maybe" for event in filtered))
    metric_cols[3].metric("Cities", len({event_city(event) for event in filtered if event_city(event)}))

    if not filtered:
        st.info("No saved shows match those calendar filters.")
        return

    payload = _calendar_payload(filtered)
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    dated = [
        pd.to_datetime(event_date(event), errors="coerce")
        for event in filtered
    ]
    valid_dates = [value for value in dated if not pd.isna(value)]
    future_dates = [value for value in valid_dates if value.date() >= date.today()]
    initial_date = min(future_dates or valid_dates).strftime("%Y-%m-%d") if valid_dates else date.today().isoformat()

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.19/index.global.min.js"></script>
  <style>
    :root {{
      --ink:#171b26;
      --muted:#737b8c;
      --line:#e5e8ef;
      --panel:#ffffff;
      --soft:#f7f8fb;
      --coral:#ff5b57;
      --blue:#4f73ff;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:transparent;
    }}
    #calendar-shell {{
      border:1px solid var(--line);
      border-radius:22px;
      background:var(--panel);
      box-shadow:0 16px 38px rgba(17,24,39,.06);
      padding:18px;
      overflow:hidden;
    }}
    .fc .fc-toolbar {{
      gap:12px;
      margin-bottom:18px;
    }}
    .fc .fc-toolbar-title {{
      font-size:1.35rem;
      font-weight:850;
      letter-spacing:-.025em;
    }}
    .fc .fc-button {{
      background:#fff!important;
      color:var(--ink)!important;
      border:1px solid var(--line)!important;
      border-radius:11px!important;
      box-shadow:none!important;
      font-size:.78rem!important;
      font-weight:800!important;
      padding:.55rem .72rem!important;
    }}
    .fc .fc-button-active,
    .fc .fc-button:hover {{
      background:var(--ink)!important;
      color:#fff!important;
      border-color:var(--ink)!important;
    }}
    .fc-theme-standard td,
    .fc-theme-standard th,
    .fc-theme-standard .fc-scrollgrid {{
      border-color:var(--line)!important;
    }}
    .fc .fc-col-header-cell-cushion {{
      padding:10px 4px;
      color:#8a92a3;
      font-size:.69rem;
      font-weight:900;
      letter-spacing:.1em;
      text-transform:uppercase;
      text-decoration:none;
    }}
    .fc .fc-daygrid-day-number {{
      padding:8px;
      color:var(--ink);
      font-weight:850;
      text-decoration:none;
    }}
    .fc .fc-day-today {{
      background:#fff8f7!important;
    }}
    .fc .fc-event {{
      border-radius:8px;
      padding:2px 4px;
      border:0;
      cursor:pointer;
      font-size:.72rem;
      font-weight:780;
      box-shadow:none;
    }}
    .fc .fc-daygrid-more-link {{
      color:var(--coral);
      font-size:.72rem;
      font-weight:850;
    }}
    .fc .fc-popover {{
      border:1px solid var(--line);
      border-radius:14px;
      box-shadow:0 18px 45px rgba(17,24,39,.16);
      overflow:hidden;
    }}
    .fc .fc-list-event:hover td {{
      background:#fff8f7!important;
    }}
    .fc .fc-list-event-title a {{
      color:var(--ink);
      font-weight:800;
      text-decoration:none;
    }}
    .legend {{
      display:flex;
      gap:16px;
      flex-wrap:wrap;
      color:var(--muted);
      font-size:.75rem;
      margin:0 0 14px;
    }}
    .legend span {{
      display:inline-flex;
      align-items:center;
      gap:6px;
    }}
    .dot {{
      width:8px;
      height:8px;
      border-radius:50%;
      display:inline-block;
    }}
    #drawer-backdrop {{
      position:fixed;
      inset:0;
      background:rgba(17,24,39,.38);
      display:none;
      align-items:stretch;
      justify-content:flex-end;
      z-index:99999;
    }}
    #drawer {{
      width:min(430px,92vw);
      height:100%;
      background:#fff;
      padding:22px;
      box-shadow:-22px 0 60px rgba(17,24,39,.22);
      transform:translateX(100%);
      transition:transform .2s ease;
      overflow:auto;
    }}
    #drawer-backdrop.open {{ display:flex; }}
    #drawer-backdrop.open #drawer {{ transform:translateX(0); }}
    .drawer-close {{
      border:1px solid var(--line);
      background:#fff;
      border-radius:999px;
      width:36px;
      height:36px;
      font-size:18px;
      cursor:pointer;
      float:right;
    }}
    .drawer-kicker {{
      color:var(--coral);
      text-transform:uppercase;
      letter-spacing:.12em;
      font-size:.69rem;
      font-weight:900;
      margin-top:48px;
    }}
    .drawer-title {{
      font-size:1.55rem;
      line-height:1.08;
      font-weight:900;
      letter-spacing:-.035em;
      margin:8px 0 8px;
    }}
    .drawer-meta {{
      color:var(--muted);
      font-size:.88rem;
      line-height:1.5;
    }}
    .drawer-score {{
      display:inline-flex;
      margin:14px 0;
      padding:7px 10px;
      border-radius:999px;
      background:#edf9f1;
      border:1px solid #c4e8ce;
      color:#24643a;
      font-size:.76rem;
      font-weight:850;
    }}
    .drawer-reason {{
      background:var(--soft);
      border:1px solid var(--line);
      border-radius:16px;
      padding:14px;
      color:#4f586b;
      line-height:1.48;
      font-size:.88rem;
      margin:0 0 16px;
    }}
    .drawer-actions {{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:9px;
    }}
    .drawer-actions a {{
      display:flex;
      justify-content:center;
      align-items:center;
      min-height:43px;
      border-radius:12px;
      border:1px solid var(--line);
      color:var(--ink);
      text-decoration:none;
      font-size:.82rem;
      font-weight:850;
      padding:8px;
      text-align:center;
    }}
    .drawer-actions a.primary {{
      background:var(--coral);
      color:#fff;
      border-color:var(--coral);
    }}
    .drawer-actions a.disabled {{
      color:#a5abba;
      pointer-events:none;
      background:#f7f8fb;
    }}
    @media(max-width:720px) {{
      #calendar-shell {{ padding:10px; border-radius:16px; }}
      .fc .fc-toolbar {{ flex-direction:column; align-items:flex-start; }}
      .fc .fc-toolbar-chunk {{ display:flex; flex-wrap:wrap; gap:5px; }}
      .fc .fc-daygrid-event {{ white-space:normal; }}
    }}
  </style>
</head>
<body>
  <div id="calendar-shell">
    <div class="legend">
      <span><i class="dot" style="background:#ff5b57"></i>Want</span>
      <span><i class="dot" style="background:#4f73ff"></i>Maybe</span>
      <span><i class="dot" style="background:#8d95a6"></i>Don't go</span>
      <span>Month view shows two events per day, then +more.</span>
    </div>
    <div id="calendar"></div>
  </div>

  <div id="drawer-backdrop" aria-hidden="true">
    <aside id="drawer" role="dialog" aria-modal="true" aria-label="Concert details">
      <button class="drawer-close" aria-label="Close">×</button>
      <div class="drawer-kicker" id="drawer-status"></div>
      <div class="drawer-title" id="drawer-title"></div>
      <div class="drawer-meta" id="drawer-meta"></div>
      <div class="drawer-score" id="drawer-score"></div>
      <div class="drawer-reason" id="drawer-reason"></div>
      <div class="drawer-actions">
        <a id="drawer-ticket" class="primary" target="_blank" rel="noopener">Tickets</a>
        <a id="drawer-plan">Plan the night</a>
        <a id="drawer-calendar" download="encore-show.ics">Add to calendar</a>
        <a id="drawer-spotify" target="_blank" rel="noopener">Open Spotify</a>
      </div>
    </aside>
  </div>

  <script id="calendar-events" type="application/json">{payload_json}</script>
  <script>
    const events = JSON.parse(document.getElementById("calendar-events").textContent);
    const backdrop = document.getElementById("drawer-backdrop");
    const closeButton = document.querySelector(".drawer-close");

    function setLink(element, url) {{
      if (url) {{
        element.href = url;
        element.classList.remove("disabled");
      }} else {{
        element.removeAttribute("href");
        element.classList.add("disabled");
      }}
    }}

    function openDrawer(info) {{
      info.jsEvent.preventDefault();
      const event = info.event;
      const props = event.extendedProps || {{}};

      document.getElementById("drawer-status").textContent = props.status || "Saved show";
      document.getElementById("drawer-title").textContent = event.title || "Concert";
      document.getElementById("drawer-meta").textContent =
        [props.venue, [props.city, props.state].filter(Boolean).join(", "), event.start?.toLocaleString()]
          .filter(Boolean).join(" · ");
      document.getElementById("drawer-score").textContent =
        props.score ? `Match ${{props.score}}` : "Saved concert";
      document.getElementById("drawer-reason").textContent =
        [props.reason, props.price].filter(Boolean).join(" · ") || "Saved in Encore AI.";

      setLink(document.getElementById("drawer-ticket"), props.ticketUrl);
      setLink(document.getElementById("drawer-spotify"), props.spotifyUrl);
      setLink(document.getElementById("drawer-calendar"), props.calendarUrl);

      const plan = document.getElementById("drawer-plan");
      plan.href = "#";
      plan.onclick = (clickEvent) => {{
        clickEvent.preventDefault();
        const url = new URL(window.top.location.href);
        url.searchParams.set("plan_event", props.planId || event.id);
        url.searchParams.set("encore_tab", "copilot");
        window.top.location.href = url.toString();
      }};

      backdrop.classList.add("open");
      backdrop.setAttribute("aria-hidden", "false");
    }}

    function closeDrawer() {{
      backdrop.classList.remove("open");
      backdrop.setAttribute("aria-hidden", "true");
    }}

    closeButton.addEventListener("click", closeDrawer);
    backdrop.addEventListener("click", (event) => {{
      if (event.target === backdrop) closeDrawer();
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape") closeDrawer();
    }});

    document.addEventListener("DOMContentLoaded", function() {{
      const calendar = new FullCalendar.Calendar(document.getElementById("calendar"), {{
        initialView: "dayGridMonth",
        initialDate: "{initial_date}",
        height: "auto",
        fixedWeekCount: false,
        navLinks: true,
        dayMaxEvents: 2,
        moreLinkClick: "popover",
        nowIndicator: true,
        eventDisplay: "block",
        headerToolbar: {{
          left: "prev,next today",
          center: "title",
          right: "dayGridMonth,listMonth"
        }},
        buttonText: {{
          today: "Today",
          month: "Month",
          list: "Agenda"
        }},
        events,
        eventClick: openDrawer
      }});
      calendar.render();
    }});
  </script>
</body>
</html>
"""

    components.html(html, height=790, scrolling=False)
