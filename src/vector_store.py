from typing import List, Dict, Any

def upsert_events_to_chroma(events: List[Dict[str, Any]], event_texts: List[str], embeddings):
    # Intentionally no-op in the fast left-sidebar build.
    # Ranking still works; we just avoid local Chroma setup/install overhead.
    return None
