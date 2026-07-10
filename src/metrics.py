from __future__ import annotations

from typing import Dict, Iterable, Optional
import numpy as np
from sklearn.metrics import ndcg_score


def precision_at_k(y_true, y_score, k=5, positive_threshold=2):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_score = np.asarray(y_score, dtype=float).flatten()
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1][:k]
    return float((y_true[order] >= positive_threshold).sum() / k)


def hit_rate_at_k(y_true, y_score, k=10, positive_threshold=2):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_score = np.asarray(y_score, dtype=float).flatten()
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1][:k]
    return float((y_true[order] >= positive_threshold).any())


def negative_rate_at_k(y_true, y_score, k=10, negative_threshold=0):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_score = np.asarray(y_score, dtype=float).flatten()
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1][:k]
    return float((y_true[order] <= negative_threshold).sum() / k)


def ndcg_at_k(y_true, y_score, k=10):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_score = np.asarray(y_score, dtype=float).flatten()
    if len(y_true) < 2 or len(y_score) < 2:
        return 0.0
    n = min(len(y_true), len(y_score))
    y_true, y_score = y_true[:n], y_score[:n]
    if np.max(y_true) <= 0:
        return 0.0
    return float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=min(k, n)))


def evaluate_ranking(y_true, baseline_scores, model_scores=None, k_values=(5, 10)):
    results = {}
    for k in k_values:
        results[f"baseline_precision_at_{k}"] = precision_at_k(y_true, baseline_scores, k=k)
        results[f"baseline_hit_rate_at_{k}"] = hit_rate_at_k(y_true, baseline_scores, k=k)
        results[f"baseline_negative_rate_at_{k}"] = negative_rate_at_k(y_true, baseline_scores, k=k)
        results[f"baseline_ndcg_at_{k}"] = ndcg_at_k(y_true, baseline_scores, k=k)
        if model_scores is not None:
            results[f"model_precision_at_{k}"] = precision_at_k(y_true, model_scores, k=k)
            results[f"model_hit_rate_at_{k}"] = hit_rate_at_k(y_true, model_scores, k=k)
            results[f"model_negative_rate_at_{k}"] = negative_rate_at_k(y_true, model_scores, k=k)
            results[f"model_ndcg_at_{k}"] = ndcg_at_k(y_true, model_scores, k=k)
    return results


def evaluate_grouped_ranking(
    y_true,
    baseline_scores,
    model_scores,
    group_ids: Iterable,
    direct_flags: Optional[Iterable] = None,
    k_values=(5, 10),
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    model_scores = np.asarray(model_scores, dtype=float)
    group_ids = np.asarray(list(group_ids))
    direct_flags = np.asarray(list(direct_flags), dtype=float) if direct_flags is not None else np.zeros(len(y_true))

    per_group = []
    for group in dict.fromkeys(group_ids.tolist()):
        mask = group_ids == group
        if mask.sum() < 2:
            continue
        row = evaluate_ranking(y_true[mask], baseline_scores[mask], model_scores[mask], k_values=k_values)
        for prefix, scores in [("baseline", baseline_scores[mask]), ("model", model_scores[mask])]:
            k = min(10, mask.sum())
            order = np.argsort(scores)[::-1][:k]
            direct_total = direct_flags[mask].sum()
            row[f"{prefix}_direct_recall_at_10"] = (
                float(direct_flags[mask][order].sum() / direct_total) if direct_total > 0 else 1.0
            )
        per_group.append(row)

    if not per_group:
        return evaluate_ranking(y_true, baseline_scores, model_scores, k_values=k_values)

    keys = sorted({key for row in per_group for key in row})
    results = {key: float(np.mean([row.get(key, 0.0) for row in per_group])) for key in keys}
    results["evaluated_groups"] = float(len(per_group))
    return results
