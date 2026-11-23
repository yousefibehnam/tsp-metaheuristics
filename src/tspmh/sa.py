"""Simulated annealing with 2-opt (segment reversal) moves and geometric cooling."""

from __future__ import annotations

import math
import time

import numpy as np

from .heuristics import Result
from .tsplib import tour_length


def initial_temperature(tour: list[int], dist: list[list[int]], rng, accept: float = 0.5) -> float:
    """Temperature at which an average uphill 2-opt move from ``tour`` is accepted with ``accept``
    probability (estimated from 1,000 random moves)."""
    n = len(tour)
    uphill = []
    for _ in range(1000):
        i, j = sorted(rng.choice(n, 2, replace=False).tolist())
        a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % n]
        delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
        if delta > 0:
            uphill.append(delta)
    mean_uphill = float(np.mean(uphill)) if uphill else 1.0
    return -mean_uphill / math.log(accept)


def anneal(
    dist: np.ndarray,
    seed: int = 0,
    evaluations: int = 200_000,
    start: np.ndarray | None = None,
    t0: float | None = None,
    final_ratio: float = 1e-3,
    records: int = 200,
) -> Result:
    """Minimise the tour length by simulated annealing.

    Each step proposes reversing a random segment ``tour[i..j]`` (a 2-opt move), whose length change
    is computed in O(1). Improvements are always accepted; a longer tour is accepted with
    probability ``exp(-delta / T)``. The temperature falls geometrically from ``t0`` (by default set
    so that an average uphill move starts out 50 % likely) to ``final_ratio * t0`` over the budget.
    """
    began = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(dist)
    d = dist.tolist()  # plain lists: scalar indexing is several times faster than on arrays
    tour = (rng.permutation(n) if start is None else np.asarray(start)).tolist()
    length = tour_length(tour, dist)
    best, best_length = tour[:], length
    temperature = initial_temperature(tour, d, rng) if t0 is None else t0
    cooling = final_ratio ** (1.0 / evaluations)
    every = max(1, evaluations // records)
    history = [(0, best_length)]

    # random numbers in blocks: far cheaper than one generator call per step
    block = 65_536
    pairs = rng.integers(0, n, size=(block, 2)).tolist()
    uniforms = rng.random(block).tolist()
    k = 0
    for step in range(1, evaluations + 1):
        if k == block:
            pairs = rng.integers(0, n, size=(block, 2)).tolist()
            uniforms = rng.random(block).tolist()
            k = 0
        i, j = pairs[k]
        u = uniforms[k]
        k += 1
        if i > j:
            i, j = j, i
        if i == j or (i == 0 and j == n - 1):
            delta = 0  # not a real move
        else:
            a, b, c, e = tour[i - 1], tour[i], tour[j], tour[(j + 1) % n]
            delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
            if delta <= 0 or u < math.exp(-delta / temperature):
                tour[i : j + 1] = tour[i : j + 1][::-1]
                length += delta
                if length < best_length:
                    best, best_length = tour[:], length
        temperature *= cooling
        if step % every == 0:
            history.append((step, best_length))

    if history[-1] != (evaluations, best_length):
        history.append((evaluations, best_length))
    best_tour = np.array(best, dtype=np.int64)
    assert tour_length(best_tour, dist) == best_length  # the O(1) bookkeeping stayed exact
    return Result("SA", best_tour, best_length, evaluations, time.perf_counter() - began, history)
