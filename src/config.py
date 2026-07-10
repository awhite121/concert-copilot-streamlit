import os
from dotenv import load_dotenv

load_dotenv()

def get_secret(name: str, default=None):
    """
    Reads from Streamlit secrets if deployed, then environment variables / local .env.
    Keeps local dev and Streamlit Cloud deployment simple.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)
