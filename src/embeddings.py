from typing import List
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

# Fast local text embeddings.
# This avoids torch / sentence-transformers / Keras conflicts while keeping semantic-ish ranking.
_VECTOR_DIM = 768

def _clean_texts(texts: List[str]) -> List[str]:
    return [(t or "").strip() for t in texts]

def embed_texts(texts: List[str]) -> np.ndarray:
    texts = _clean_texts(texts)
    vectorizer = HashingVectorizer(
        n_features=_VECTOR_DIM,
        alternate_sign=False,
        norm=None,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
    )
    matrix = vectorizer.transform(texts)
    matrix = normalize(matrix, norm="l2", copy=False)
    return matrix.astype("float32").toarray()
