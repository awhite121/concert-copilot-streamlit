# Encore AI Architecture

## High-level flow

```text
Spotify Login
    ↓
Spotify API
    ↓
Taste Profile
    ↓
Event APIs
    ↓
Recommendation Engine
    ↓
Ranked Concert Cards
    ↓
Want / Maybe / Not a fit
    ↓
Feedback Database + Model History
    ↓
Better Recommendations
    ↓
Copilot + Plan a Night
```

## Core layers

### Spotify taste layer

Uses Spotify OAuth and the Spotify Web API to pull top artists, top tracks, genres, popularity, and listening signals.

### Event data layer

Pulls upcoming concerts by city, state, radius, date range, venue, artist, and genre.

### Ranking layer

Scores each show using direct artist matches, genre fit, Spotify taste, saved feedback, model score, and recommendation style.

### Feedback loop

Want, Maybe, and Not a fit actions become labeled feedback. The trained model learns patterns from those decisions and is safely blended with Spotify ranking.

### Copilot layer

Uses ranked concerts, saved feedback, venue/time data, and nearby places to explain tradeoffs and help plan the night.
