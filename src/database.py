import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
from .config import get_secret

PREFERENCE_ACTIONS = ("want_to_go", "maybe", "not_for_me")


def db_path() -> Path:
    path = Path(get_secret("FEEDBACK_DB_PATH", "./data/feedback.sqlite"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_display_name TEXT,
                session_id TEXT NOT NULL,
                rank_position INTEGER,
                event_id TEXT,
                event_name TEXT,
                event_date TEXT,
                event_time TEXT,
                venue TEXT,
                city TEXT,
                state TEXT,
                genre TEXT,
                subgenre TEXT,
                artists_json TEXT,
                source TEXT,
                sources_json TEXT,
                url TEXT,
                image_url TEXT,
                min_price REAL,
                max_price REAL,
                price_source TEXT,
                match_confidence TEXT,
                why_recommended TEXT,
                action TEXT NOT NULL,
                label REAL,
                exact_artist_score REAL,
                has_direct_artist_match INTEGER,
                genre_overlap_count REAL,
                genre_score REAL,
                known_price REAL,
                min_price_filled REAL,
                price_score REAL,
                weekend_event REAL,
                days_until_event REAL,
                days_score REAL,
                embedding_similarity REAL,
                embedding_score REAL,
                embedding_rank_score REAL,
                source_count_score REAL,
                artist_popularity_signal REAL,
                hybrid_score REAL,
                model_score REAL,
                final_score REAL
            )
            """
        )

        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(interactions)").fetchall()}
        upgrades = {
            "embedding_rank_score": "REAL",
            "source_count_score": "REAL",
            "artist_popularity_signal": "REAL",
            "event_time": "TEXT",
            "source": "TEXT",
            "sources_json": "TEXT",
            "image_url": "TEXT",
            "min_price": "REAL",
            "max_price": "REAL",
            "price_source": "TEXT",
            "match_confidence": "TEXT",
            "why_recommended": "TEXT",
            "feedback_reason": "TEXT",
            "winning_genre_cluster": "TEXT",
            "winning_genre_cluster_label": "TEXT",
            "anchor_artists_json": "TEXT",
            "direct_artist_rank_score": "REAL",
            "track_affinity_score": "REAL",
            "spotify_durability_score": "REAL",
            "artist_blend_score": "REAL",
            "user_cluster_affinity": "REAL",
            "event_cluster_confidence": "REAL",
            "genre_cluster_score": "REAL",
            "known_starting_price": "REAL",
            "known_typical_price": "REAL",
            "price_under_50": "REAL",
            "price_50_100": "REAL",
            "price_100_175": "REAL",
            "price_over_175": "REAL",
            "friday_event": "REAL",
            "saturday_event": "REAL",
            "weekday_index": "REAL",
            "event_hour": "REAL",
            "evening_event": "REAL",
            "known_event_time": "REAL",
            "has_multiple_sources": "REAL",
            "listing_count_signal": "REAL",
            "venue_quality_signal": "REAL",
            "discovery_quality_score": "REAL",
            "familiarity_score": "REAL",
            "novelty_score": "REAL",
            "cluster_country_americana": "REAL",
            "cluster_folk_singer_songwriter": "REAL",
            "cluster_hip_hop_rap": "REAL",
            "cluster_electronic_dance": "REAL",
            "cluster_indie_alternative": "REAL",
            "cluster_rock_punk_metal": "REAL",
            "cluster_pop": "REAL",
            "cluster_rnb_soul_funk": "REAL",
            "cluster_latin": "REAL",
            "cluster_jazz_blues": "REAL",
            "cluster_classical_instrumental": "REAL",
        }
        for col, ddl in upgrades.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE interactions ADD COLUMN {col} {ddl}")

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interactions_user_session
            ON interactions(user_id, session_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interactions_event_action
            ON interactions(event_id, action)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interactions_user_event_action
            ON interactions(user_id, event_id, action)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                task TEXT,
                model TEXT,
                used_llm INTEGER,
                latency_ms INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                error TEXT
            )
            """
        )


def delete_existing_preference(user_id: str, event_id: str):
    init_db()
    if not event_id:
        return
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM interactions
            WHERE user_id = ?
              AND event_id = ?
              AND action IN ('want_to_go', 'maybe', 'not_for_me')
            """,
            (user_id, event_id),
        )


def delete_preference(user_id: str, event_id: str):
    delete_existing_preference(user_id, event_id)


def insert_interaction(row: dict):
    init_db()
    columns = [
        "created_at", "user_id", "user_display_name", "session_id", "rank_position",
        "event_id", "event_name", "event_date", "event_time", "venue", "city", "state", "genre", "subgenre",
        "artists_json", "source", "sources_json", "url", "image_url", "min_price", "max_price", "price_source",
        "match_confidence", "why_recommended", "feedback_reason", "action", "label",
        "winning_genre_cluster", "winning_genre_cluster_label", "anchor_artists_json",
        "exact_artist_score", "has_direct_artist_match", "direct_artist_rank_score",
        "track_affinity_score", "spotify_durability_score", "artist_blend_score",
        "genre_overlap_count", "genre_score", "user_cluster_affinity",
        "event_cluster_confidence", "genre_cluster_score",
        "known_price", "known_starting_price", "known_typical_price", "min_price_filled", "price_score",
        "price_under_50", "price_50_100", "price_100_175", "price_over_175",
        "weekend_event", "friday_event", "saturday_event", "weekday_index", "event_hour",
        "evening_event", "known_event_time", "days_until_event", "days_score",
        "embedding_similarity", "embedding_score", "embedding_rank_score",
        "source_count_score", "has_multiple_sources", "listing_count_signal",
        "artist_popularity_signal", "venue_quality_signal", "discovery_quality_score",
        "familiarity_score", "novelty_score",
        "cluster_country_americana", "cluster_folk_singer_songwriter", "cluster_hip_hop_rap",
        "cluster_electronic_dance", "cluster_indie_alternative", "cluster_rock_punk_metal",
        "cluster_pop", "cluster_rnb_soul_funk", "cluster_latin", "cluster_jazz_blues",
        "cluster_classical_instrumental", "hybrid_score", "model_score", "final_score",
    ]
    payload = {col: row.get(col) for col in columns}
    payload["created_at"] = payload["created_at"] or datetime.utcnow().isoformat()
    placeholders = ",".join(["?"] * len(columns))
    sql = f"INSERT INTO interactions ({','.join(columns)}) VALUES ({placeholders})"
    with get_connection() as conn:
        conn.execute(sql, [payload[col] for col in columns])


def load_interactions(include_shown: bool = True) -> pd.DataFrame:
    init_db()
    query = "SELECT * FROM interactions"
    if not include_shown:
        query += " WHERE action != 'shown' AND action != 'opened_tickets'"
    query += " ORDER BY created_at DESC"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def load_latest_preferences() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM interactions
            WHERE action IN ('want_to_go', 'maybe', 'not_for_me')
              AND label IS NOT NULL
            ORDER BY created_at DESC
            """,
            conn,
        )
    if df.empty:
        return df
    return df.drop_duplicates(subset=["user_id", "event_id"], keep="first").copy()


def load_user_latest_preferences(user_id: str) -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM interactions
            WHERE user_id = ?
              AND action IN ('want_to_go', 'maybe', 'not_for_me')
              AND label IS NOT NULL
            ORDER BY created_at DESC
            """,
            conn,
            params=(user_id,),
        )
    if df.empty:
        return df
    return df.drop_duplicates(subset=["event_id"], keep="first").copy()


def load_user_opened_tickets(user_id: str) -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM interactions
            WHERE user_id = ? AND action = 'opened_tickets'
            ORDER BY created_at DESC
            """,
            conn,
            params=(user_id,),
        )
    if df.empty:
        return df
    return df.drop_duplicates(subset=["event_id"], keep="first").copy()


def clear_all_feedback():
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM interactions")


def insert_llm_call(task: str, result):
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO llm_calls (
                created_at, task, model, used_llm, latency_ms,
                input_tokens, output_tokens, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                task,
                getattr(result, "model", None),
                1 if getattr(result, "used_llm", False) else 0,
                getattr(result, "latency_ms", None),
                getattr(result, "input_tokens", None),
                getattr(result, "output_tokens", None),
                getattr(result, "error", None),
            ),
        )


def load_llm_calls() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM llm_calls ORDER BY created_at DESC", conn)
