"""
Unit tests for ranking metrics, using small hand-computed examples so the
expected values can be verified by hand -- run with `pytest tests/`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.metrics import catalog_coverage, map_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_at_k_perfect():
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 2, 3}
    assert recall_at_k(recommended, relevant, k=5) == 1.0


def test_recall_at_k_partial():
    recommended = [1, 6, 7, 8, 9]
    relevant = {1, 2, 3}
    assert recall_at_k(recommended, relevant, k=5) == pytest_approx(1 / 3)


def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_precision_at_k():
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 2}
    assert precision_at_k(recommended, relevant, k=5) == pytest_approx(2 / 5)


def test_ndcg_rewards_early_hits():
    relevant = {1, 2}
    high_first = [1, 2, 3, 4, 5]
    high_last = [3, 4, 5, 1, 2]
    assert ndcg_at_k(high_first, relevant, k=5) > ndcg_at_k(high_last, relevant, k=5)


def test_ndcg_perfect_ordering_equals_one():
    relevant = {1, 2}
    recommended = [1, 2, 3, 4, 5]
    assert ndcg_at_k(recommended, relevant, k=2) == pytest_approx(1.0)


def test_map_at_k_perfect():
    recommended = [1, 2, 3]
    relevant = {1, 2, 3}
    assert map_at_k(recommended, relevant, k=3) == pytest_approx(1.0)


def test_catalog_coverage():
    lists = [[1, 2], [2, 3], [1]]
    assert catalog_coverage(lists, catalog_size=10) == pytest_approx(3 / 10)


def pytest_approx(value, tol=1e-6):
    """Tiny local helper so this file has zero extra dependencies beyond pytest itself."""

    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

    return _Approx()
