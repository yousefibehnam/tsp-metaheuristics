"""Construction and local-search baselines: nearest neighbour and 2-opt."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .tsplib import tour_length


@dataclass
class Result:
    """Outcome of one run: the best tour, its length and the best length seen over time."""

    algorithm: str
    tour: np.ndarray
    length: int
    evaluations: int = 0  # candidate tours whose length was computed or updated
    seconds: float = 0.0
    history: list[tuple[int, int]] = field(default_factory=list)  # (evaluations, best length)


def nearest_neighbour(dist: np.ndarray, start: int = 0) -> Result:
    """Greedy tour: from ``start``, always go to the closest unvisited city (ties: lowest index)."""
    began = time.perf_counter()
    n = len(dist)
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int64)
    city = start
    for step in range(n):
        tour[step] = city
        visited[city] = True
        if step < n - 1:
            row = np.where(visited, np.iinfo(np.int64).max, dist[city])
            city = int(np.argmin(row))
    length = tour_length(tour, dist)
    return Result("NN", tour, length, n, time.perf_counter() - began, [(n, length)])


def two_opt_delta(tour, dist, i: int, j: int) -> int:
    """Length change from reversing ``tour[i..j]``, in O(1). Needs 0 <= i < j < n, except the
    pair i = 0, j = n - 1, which reverses the whole tour and is no move at all.

    The tour edges (a, b) and (c, d) with a = tour[i-1], b = tour[i], c = tour[j], d = tour[j+1]
    are replaced by (a, c) and (b, d).
    """
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    return int(dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d])


def two_opt(tour: np.ndarray, dist: np.ndarray) -> Result:
    """Best-improvement 2-opt local search until no reversal shortens the tour."""
    began = time.perf_counter()
    tour = np.array(tour, dtype=np.int64)
    n = len(tour)
    evaluations = 0
    improved = True
    while improved:
        improved = False
        best_delta, best_i, best_j = 0, -1, -1
        for i in range(n - 1):
            # j from i + 1 to n - 1; reversing the whole tour (i = 0, j = n - 1) changes nothing
            j = np.arange(i + 1, n - 1 if i == 0 else n)
            a, b = tour[i - 1], tour[i]
            c, d = tour[j], tour[(j + 1) % n]
            delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
            evaluations += len(j)
            k = int(np.argmin(delta))
            if delta[k] < best_delta:
                best_delta, best_i, best_j = int(delta[k]), i, int(j[k])
        if best_delta < 0:
            tour[best_i : best_j + 1] = tour[best_i : best_j + 1][::-1]
            improved = True
    length = tour_length(tour, dist)
    return Result("2-opt", tour, length, evaluations, time.perf_counter() - began)
