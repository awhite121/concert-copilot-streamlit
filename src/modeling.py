from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil

import joblib
import numpy as np
import pandas as pd

from .config import get_secret
from .feedback import get_labeled_feedback_df
from .features import BASE_NUMERIC_COLS
from .metrics import evaluate_grouped_ranking


FEATURE_COLS = [column for column in BASE_NUMERIC_COLS if column != "hybrid_score"] + ["hybrid_score"]


def model_path() -> Path:
    path = Path(get_secret("MODEL_PATH", "./models/xgb_feedback_ranker.joblib"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def previous_model_path() -> Path:
    path = model_path()
    return path.with_name(path.stem + "_previous" + path.suffix)


def history_dir() -> Path:
    path = model_path().parent / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in FEATURE_COLS + ["label"]:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    if "session_id" not in out.columns:
        out["session_id"] = "session_0"
    out["session_id"] = out["session_id"].fillna("unknown_session").astype(str)
    # Learning-to-rank needs at least two rated choices in a query/search session.
    valid_sessions = out.groupby("session_id").size()
    valid_sessions = set(valid_sessions[valid_sessions >= 2].index)
    return out[out["session_id"].isin(valid_sessions)].dropna(subset=["label"]).copy()


def split_train_test_by_session(df: pd.DataFrame, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    sessions = df["session_id"].dropna().unique().tolist()
    rng.shuffle(sessions)
    if len(sessions) >= 4:
        n_test = max(2, int(round(len(sessions) * test_size)))
        n_test = min(n_test, len(sessions) - 1)
        test_sessions = set(sessions[:n_test])
        train_df = df[~df["session_id"].isin(test_sessions)].copy()
        test_df = df[df["session_id"].isin(test_sessions)].copy()
        return train_df, test_df

    # Early-stage fallback: preserve groups by putting at least one session in holdout.
    if len(sessions) >= 2:
        test_sessions = {sessions[0]}
        return (
            df[~df["session_id"].isin(test_sessions)].copy(),
            df[df["session_id"].isin(test_sessions)].copy(),
        )

    # One session can train a candidate for inspection, but cannot be safely promoted.
    ordered = df.sort_values(["session_id", "created_at"] if "created_at" in df.columns else ["session_id"])
    split = max(2, int(len(ordered) * 0.75))
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def _numeric_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    output = df.copy()
    for column in feature_cols:
        if column not in output.columns:
            output[column] = 0.0
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    return output[feature_cols]


def _normalize_predictions(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).flatten()
    if len(values) == 0:
        return values
    if np.nanmax(values) == np.nanmin(values):
        return np.full(len(values), 50.0)
    return pd.Series(values).rank(pct=True, method="average").to_numpy() * 100.0


def _predict_bundle(bundle: Dict[str, Any], feature_df: pd.DataFrame, normalized: bool = False) -> np.ndarray:
    model = bundle["model"]
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)
    X = _numeric_frame(feature_df, feature_cols)
    predictions = np.asarray(model.predict(X), dtype=float)
    if normalized:
        return _normalize_predictions(predictions)
    return predictions


def _recommended_weight(n_rows: int) -> float:
    if n_rows < 100:
        return 0.15
    if n_rows < 250:
        return 0.20
    if n_rows < 500:
        return 0.25
    return 0.30


def _metric(metrics: Dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _comparison_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "precision5": _metric(metrics, f"{prefix}_precision_at_5"),
        "ndcg5": _metric(metrics, f"{prefix}_ndcg_at_5"),
        "ndcg10": _metric(metrics, f"{prefix}_ndcg_at_10"),
        "negative10": _metric(metrics, f"{prefix}_negative_rate_at_10"),
        "direct10": _metric(metrics, f"{prefix}_direct_recall_at_10"),
    }


def _promotion_decision(
    candidate_metrics: Dict[str, Any],
    current_metrics: Optional[Dict[str, Any]],
    test_rows: int,
    test_groups: int,
) -> Tuple[bool, str, float, str]:
    baseline = _comparison_metrics("baseline", candidate_metrics)
    candidate = _comparison_metrics("model", candidate_metrics)
    comparator_name = "cold-start baseline"
    comparator = baseline

    if current_metrics:
        current = _comparison_metrics("model", current_metrics)
        # Use whichever comparator has the stronger NDCG@10. A candidate must not
        # replace a good deployed model merely because it beats the original baseline.
        if current["ndcg10"] >= comparator["ndcg10"]:
            comparator, comparator_name = current, "current promoted model"

    average_delta = np.mean([
        candidate["precision5"] - comparator["precision5"],
        candidate["ndcg5"] - comparator["ndcg5"],
        candidate["ndcg10"] - comparator["ndcg10"],
    ])

    if test_rows < 15 or test_groups < 2:
        return (
            False,
            "Candidate saved, but automatic promotion requires at least 15 holdout ratings across 2 search sessions.",
            float(average_delta),
            comparator_name,
        )
    if candidate["ndcg10"] < comparator["ndcg10"] - 0.005:
        return False, f"Not promoted: NDCG@10 was below the {comparator_name}.", float(average_delta), comparator_name
    if candidate["ndcg5"] < comparator["ndcg5"] - 0.02:
        return False, f"Not promoted: top-5 ranking quality declined versus the {comparator_name}.", float(average_delta), comparator_name
    if candidate["negative10"] > comparator["negative10"] + 0.05:
        return False, "Not promoted: too many Not-for-Me events moved into the top 10.", float(average_delta), comparator_name
    if candidate["direct10"] < comparator["direct10"] - 0.10:
        return False, "Not promoted: it pushed too many direct Spotify matches out of the top 10.", float(average_delta), comparator_name

    meaningful_win = candidate["ndcg10"] >= comparator["ndcg10"] + 0.01 or average_delta >= 0.015
    if not meaningful_win:
        return False, f"Candidate was safe but did not meaningfully improve on the {comparator_name}.", float(average_delta), comparator_name

    return True, f"Promoted: it safely improved ranking versus the {comparator_name}.", float(average_delta), comparator_name


def _fit_candidate(train_df: pd.DataFrame):
    backend_warning = None
    try:
        from xgboost import XGBRanker

        model = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            n_estimators=220,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.82,
            min_child_weight=3,
            reg_alpha=0.20,
            reg_lambda=4.0,
            random_state=42,
            n_jobs=2,
            tree_method="hist",
        )
        ordered = train_df.sort_values("session_id").copy()
        qid = pd.factorize(ordered["session_id"])[0]
        model.fit(_numeric_frame(ordered, FEATURE_COLS), ordered["label"].astype(float), qid=qid)
        return model, "xgboost_learning_to_rank", backend_warning
    except Exception as exc:
        from sklearn.ensemble import RandomForestRegressor

        backend_warning = f"XGBoost ranker unavailable ({exc}). Used RandomForest fallback; candidate can be reviewed but is held to the same guardrails."
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(_numeric_frame(train_df, FEATURE_COLS), train_df["label"].astype(float))
        return model, "sklearn_random_forest_fallback", backend_warning


def train_feedback_model(min_rows: int = 60) -> Dict[str, Any]:
    raw = get_labeled_feedback_df()
    df = prepare_training_frame(raw)
    session_count = int(df["session_id"].nunique()) if not df.empty else 0

    if len(df) < min_rows:
        return {
            "ok": False,
            "message": f"Need at least {min_rows} usable ratings across search sessions. Current usable ratings: {len(df)}.",
            "rows": len(df),
            "sessions": session_count,
        }
    if session_count < 2:
        return {"ok": False, "message": "Rate concerts in at least two recommendation searches before training.", "rows": len(df)}
    if df["label"].nunique() < 2:
        return {"ok": False, "message": "Need at least two preference values, such as positive and Not for Me.", "rows": len(df)}

    train_df, test_df = split_train_test_by_session(df)
    if train_df.empty or test_df.empty:
        return {"ok": False, "message": "Not enough grouped train/test data after splitting.", "rows": len(df)}

    model, backend, backend_warning = _fit_candidate(train_df)
    candidate_bundle = {"model": model, "feature_cols": FEATURE_COLS, "model_type": backend}
    train_pred = _predict_bundle(candidate_bundle, train_df)
    test_pred = _predict_bundle(candidate_bundle, test_df)

    train_metrics = evaluate_grouped_ranking(
        train_df["label"].values,
        train_df["hybrid_score"].values,
        train_pred,
        train_df["session_id"].values,
        train_df.get("has_direct_artist_match", pd.Series(np.zeros(len(train_df)))).values,
    )
    test_metrics = evaluate_grouped_ranking(
        test_df["label"].values,
        test_df["hybrid_score"].values,
        test_pred,
        test_df["session_id"].values,
        test_df.get("has_direct_artist_match", pd.Series(np.zeros(len(test_df)))).values,
    )

    current_bundle = load_feedback_model("current")
    current_metrics = None
    if current_bundle is not None:
        try:
            current_pred = _predict_bundle(current_bundle, test_df)
            current_metrics = evaluate_grouped_ranking(
                test_df["label"].values,
                test_df["hybrid_score"].values,
                current_pred,
                test_df["session_id"].values,
                test_df.get("has_direct_artist_match", pd.Series(np.zeros(len(test_df)))).values,
            )
        except Exception:
            current_metrics = None

    test_groups = int(test_df["session_id"].nunique())
    promoted, promotion_reason, average_delta, comparator_name = _promotion_decision(
        test_metrics,
        current_metrics,
        len(test_df),
        test_groups,
    )

    trained_at = datetime.utcnow().isoformat()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    importance = getattr(model, "feature_importances_", np.zeros(len(FEATURE_COLS)))
    feature_importance = pd.DataFrame({"feature": FEATURE_COLS, "importance": importance}).sort_values("importance", ascending=False)

    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "model_type": backend,
        "trained_at": trained_at,
        "n_rows": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_sessions": session_count,
        "n_test_sessions": test_groups,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "current_comparison_metrics": current_metrics,
        "feature_importance": feature_importance.to_dict(orient="records"),
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "promotion_comparator": comparator_name,
        "average_metric_delta": average_delta,
        "training_backend": backend,
        "backend_warning": backend_warning,
        "recommended_model_weight": _recommended_weight(len(df)),
    }

    candidate_path = history_dir() / f"ranker_{timestamp}.joblib"
    joblib.dump(bundle, candidate_path)

    current_path = model_path()
    previous_path = previous_model_path()
    if promoted:
        if current_path.exists():
            shutil.copy2(current_path, previous_path)
        joblib.dump(bundle, current_path)

    return {
        "ok": True,
        "message": "Learning-to-rank candidate trained. " + promotion_reason,
        "candidate_path": str(candidate_path),
        **{key: value for key, value in bundle.items() if key != "model"},
    }


def _safe_load_bundle(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        return bundle if isinstance(bundle, dict) and bundle.get("model") is not None else None
    except Exception:
        return None


def _safe_load_bundle(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        return bundle if isinstance(bundle, dict) and bundle.get("model") is not None else None
    except Exception:
        return None


def load_feedback_model(variant: str = "current") -> Optional[Dict[str, Any]]:
    if variant == "previous":
        return _safe_load_bundle(previous_model_path())

    configured = model_path()
    candidates = [
        configured,
        configured.parent / "xgb_feedback_reranker.joblib",
        configured.parent / "xgb_feedback_ranker.joblib",
    ]
    candidates.extend(history_dir().glob("*.joblib"))

    choices = []
    seen = set()

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)

        bundle = _safe_load_bundle(candidate)
        if bundle is None:
            continue

        is_current_file = candidate.name in {
            configured.name,
            "xgb_feedback_reranker.joblib",
            "xgb_feedback_ranker.joblib",
        }
        if not (bool(bundle.get("promoted")) or is_current_file):
            continue

        choices.append((
            int(bundle.get("n_rows") or 0),
            str(bundle.get("trained_at") or ""),
            candidate.stat().st_mtime,
            bundle,
        ))

    if not choices:
        return None

    choices.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return choices[0][3]

def predict_feedback_scores(feature_df: pd.DataFrame, variant: str = "current") -> np.ndarray:
    bundle = load_feedback_model(variant)
    if bundle is None:
        return np.asarray(feature_df.get("hybrid_score", np.zeros(len(feature_df))), dtype=float)
    return _predict_bundle(bundle, feature_df, normalized=True)


def compare_model_variants(feature_df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=feature_df.index)
    output["baseline_score"] = pd.to_numeric(feature_df.get("hybrid_score", 0), errors="coerce").fillna(0)
    output["baseline_rank"] = output["baseline_score"].rank(ascending=False, method="min").astype(int)
    for variant in ("current", "previous"):
        bundle = load_feedback_model(variant)
        if bundle is None:
            output[f"{variant}_score"] = np.nan
            output[f"{variant}_rank"] = np.nan
            continue
        scores = _predict_bundle(bundle, feature_df, normalized=True)
        output[f"{variant}_score"] = scores
        output[f"{variant}_rank"] = pd.Series(scores).rank(ascending=False, method="min").astype(int).values
    return output


def rollback_to_previous() -> Dict[str, Any]:
    current = model_path()
    previous = previous_model_path()
    if not previous.exists():
        return {"ok": False, "message": "No previous promoted model is available."}
    if current.exists():
        backup = history_dir() / f"rollback_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
        shutil.copy2(current, backup)
    shutil.copy2(previous, current)
    return {"ok": True, "message": "Previous promoted model restored as current."}


def list_model_versions(limit: int = 20) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(history_dir().glob("*.joblib"), key=lambda value: value.stat().st_mtime, reverse=True)[:limit]:
        try:
            bundle = joblib.load(path)
            metrics = bundle.get("test_metrics", {})
            rows.append({
                "file": path.name,
                "trained_at": bundle.get("trained_at"),
                "rows": bundle.get("n_rows"),
                "sessions": bundle.get("n_sessions"),
                "test_rows": bundle.get("n_test"),
                "backend": bundle.get("training_backend"),
                "promoted": bundle.get("promoted"),
                "comparator": bundle.get("promotion_comparator"),
                "avg_metric_delta": bundle.get("average_metric_delta"),
                "model_ndcg_at_10": metrics.get("model_ndcg_at_10"),
                "baseline_ndcg_at_10": metrics.get("baseline_ndcg_at_10"),
                "model_negative_rate_at_10": metrics.get("model_negative_rate_at_10"),
            })
        except Exception:
            continue
    return rows
