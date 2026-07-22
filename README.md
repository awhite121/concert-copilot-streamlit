# Encore AI

Spotify-powered concert discovery and planning assistant.

EncoreAI connects to Spotify, learns your music taste, finds upcoming concerts by city/date/radius, ranks shows with a personalized match score, learns from Want / Maybe / Not a fit feedback, and helps plan the night around a concert.

## Live App

**Streamlit:** https://encore-ai.streamlit.app/

> Spotify currently restricts OAuth access to approved users. The app supports Spotify login for approved users, and includes Demo Mode so anyone can explore the recommendation, ranking, feedback, and AI Copilot experience without connecting a Spotify account.

## What it does

- Spotify OAuth login for approved users
- Demo Mode for public access without Spotify approval
- Spotify taste profile from top artists, tracks, and genres
- Live concert search by city, date, radius, artist, genre, and venue
- Personalized scoring and feedback-based ranking
- Saved Want / Maybe / Not a fit memory
- My Playlist
- AI Copilot to compare concerts and explain recommendations
- Plan a Night workflow around selected shows
- Model history and feedback evaluation

## Product flow

```text
Spotify taste profile or Demo Mode profile
        ↓
Live event inventory
        ↓
Personalized scoring + feedback ranking
        ↓
Saved shortlist + playlist context
        ↓
Grounded AI Copilot and Plan a Night assistant
```

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

## Author

**Andrew White**  
Portfolio: https://www.andrewwhitedata.com  
GitHub: https://github.com/awhite121
