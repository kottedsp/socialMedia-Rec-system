"""
Standard ranking/retrieval metrics for recommender systems.

These are deliberately written from scratch (not imported from a library)
because being able to explain Recall@K vs NDCG@K vs MAP@K from first
principles, and why you'd report all three rather than one, is exactly the
kind of thing that comes up in an ML interview.
"""

import math
from typing import Sequence


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Of all the items the user actually engaged with, what fraction did
    we manage to surface in our top-K? Penalizes missing relevant items;
    indifferent to their exact rank within the top-K."""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Of the K items we showed, what fraction were actually relevant?
    Precision matters separately from recall because showing 50 items to
    find 2 good ones wastes feed real estate even if recall is "good"."""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Like recall, but rewards putting relevant items near the *top* of
    the list more than near the bottom of it -- this is the metric that
    actually reflects feed quality, since most users never scroll past the
    first handful of items."""

    def dcg(items):
        return sum(1.0 / math.log2(idx + 2) for idx, item in enumerate(items) if item in relevant)

    actual_dcg = dcg(recommended[:k])
    ideal_dcg = dcg(list(relevant)[:k])  # best possible ordering
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def map_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Mean Average Precision: like NDCG, rewards relevant items appearing
    early, but using precision at each rank rather than a log discount."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def catalog_coverage(all_recommended_lists: Sequence[Sequence[int]], catalog_size: int) -> float:
    """What fraction of the *entire catalog* ever gets recommended to
    anyone? Low coverage is a warning sign of a feedback loop / filter
    bubble where the model only ever surfaces the same popular items."""
    recommended_union = set()
    for rec_list in all_recommended_lists:
        recommended_union.update(rec_list)
    return len(recommended_union) / catalog_size


def intra_list_diversity(item_embeddings, recommended: Sequence[int]) -> float:
    """Average pairwise cosine distance between recommended items'
    embeddings. Low diversity means every recommendation looks the same --
    fine for precision, bad for user experience and bad for breaking out of
    filter bubbles."""
    import itertools

    import numpy as np

    if len(recommended) < 2:
        return 0.0
    vecs = [item_embeddings[i] for i in recommended]
    dists = []
    for a, b in itertools.combinations(vecs, 2):
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        dists.append(1 - cos_sim)
    return float(np.mean(dists))
