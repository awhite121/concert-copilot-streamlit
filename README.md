# Encore AI

Spotify-powered concert discovery and planning assistant.

Encore AI connects to Spotify, learns your music taste, finds upcoming concerts by city/date, ranks shows with a personalized match score, learns from Want / Maybe / Not a fit feedback, and helps plan the night around a concert.

## Streamlit Cloud setup

Main file:

```text
app.py
```

Use Streamlit Cloud secrets. Do not commit `.env`.

Example secrets:

```toml
SPOTIFY_CLIENT_ID = "..."
SPOTIFY_CLIENT_SECRET = "..."
SPOTIFY_REDIRECT_URI = "https://YOUR-APP-NAME.streamlit.app"

TICKETMASTER_API_KEY = "..."
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-5.4-mini"
SEATGEEK_CLIENT_ID = "..."
GOOGLE_MAPS_API_KEY = "..."
YELP_API_KEY = "..."
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does

- Spotify OAuth login
- Spotify taste profile from top artists, tracks, and genres
- Live concert search by city, date, radius, artist, genre, and venue
- Personalized scoring and feedback-based ranking
- Saved Want / Maybe / Not a fit memory
- My Playlist
- Copilot and Plan a Night
- Model history and feedback evaluation
