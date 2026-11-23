"""Genetic algorithm with order crossover (OX) and 2-opt mutation."""

from __future__ import annotations

import time

import numpy as np

from .heuristics import Result


def order_crossover(p1: np.ndarray, p2: np.ndarray, a: int, b: int) -> np.ndarray:
    """OX (Davis, 1985): copy ``p1[a:b]``, then fill the other positions, starting at ``b`` and
    wrapping around, with the remaining cities in the order they appear in ``p2`` from ``b`` on."""
    n = len(p1)
    child = np.empty(n, dtype=p1.dtype)
    child[a:b] = p1[a:b]
    in_segment = np.zeros(n, dtype=bool)
    in_segment[p1[a:b]] = True
    order = np.roll(p2, -b)
    rest = order[~in_segment[order]]
    child[(b + np.arange(n - (b - a))) % n] = rest
    return child


def two_opt_mutation(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Reverse ``tour[i..j]``: a random 2-opt move (two edges replaced), not a local search."""
    tour = tour.copy()
    tour[i : j + 1] = tour[i : j + 1][::-1]
    return tour


def lengths(population: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return dist[population, np.roll(population, -1, axis=1)].sum(axis=1)


def evolve(
    dist: np.ndarray,
    seed: int = 0,
    evaluations: int = 200_000,
    population: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    tournament: int = 3,
    elites: int = 2,
    records: int = 200,
) -> Result:
    """Generational GA: tournament selection, OX, 2-opt mutation and elitism.

    The budget counts tours evaluated: the initial population plus every child. Each generation
    keeps the ``elites`` best tours and breeds ``population - elites`` children.
    """
    began = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(dist)
    pop = np.array([rng.permutation(n) for _ in range(population)])
    fit = lengths(pop, dist)
    used = population
    children_per_generation = population - elites
    best_i = int(np.argmin(fit))
    best, best_length = pop[best_i].copy(), int(fit[best_i])
    history = [(used, best_length)]
    every = max(1, evaluations // records)
    next_record = (used // every + 1) * every

    def select() -> np.ndarray:
        contestants = rng.integers(0, population, tournament)
        return pop[contestants[np.argmin(fit[contestants])]]

    while used + children_per_generation <= evaluations:
        children = np.empty((children_per_generation, n), dtype=pop.dtype)
        for c in range(children_per_generation):
            p1, p2 = select(), select()
            if rng.random() < crossover_rate:
                a, b = np.sort(rng.choice(n + 1, 2, replace=False))
                child = order_crossover(p1, p2, a, b)
            else:
                child = p1.copy()
            if rng.random() < mutation_rate:
                i, j = np.sort(rng.choice(n, 2, replace=False))
                child = two_opt_mutation(child, i, j)
            children[c] = child
        child_fit = lengths(children, dist)
        used += children_per_generation
        keep = np.argsort(fit)[:elites]
        pop = np.concatenate([pop[keep], children])
        fit = np.concatenate([fit[keep], child_fit])
        i = int(np.argmin(fit))
        if fit[i] < best_length:
            best, best_length = pop[i].copy(), int(fit[i])
        if used >= next_record:
            history.append((used, best_length))
            next_record = (used // every + 1) * every

    if history[-1] != (used, best_length):
        history.append((used, best_length))
    return Result("GA", best, best_length, used, time.perf_counter() - began, history)
