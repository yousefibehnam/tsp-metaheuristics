import itertools

import numpy as np
import pytest

from tspmh.ga import evolve, lengths, order_crossover, two_opt_mutation
from tspmh.heuristics import nearest_neighbour, two_opt, two_opt_delta
from tspmh.sa import anneal
from tspmh.tsplib import distance_matrix, tour_length


def circle(n: int = 12, radius: float = 1000.0) -> np.ndarray:
    """Cities on a circle: the optimal tour visits them in angular order."""
    angles = np.arange(n) * 2 * np.pi / n
    return distance_matrix(np.c_[radius * np.cos(angles), radius * np.sin(angles)])


def random_instance(n: int, seed: int = 0) -> np.ndarray:
    return distance_matrix(np.random.default_rng(seed).uniform(0, 1000, (n, 2)))


def brute_force_optimum(dist: np.ndarray) -> int:
    n = len(dist)
    return min(tour_length([0, *p], dist) for p in itertools.permutations(range(1, n)))


def is_tour(tour, n) -> bool:
    return sorted(np.asarray(tour).tolist()) == list(range(n))


CIRCLE_OPT = tour_length(np.arange(12), circle())


def test_nearest_neighbour_on_a_circle():
    result = nearest_neighbour(circle(), start=3)
    assert result.tour[0] == 3 and is_tour(result.tour, 12)
    assert result.length == CIRCLE_OPT


def test_two_opt_delta_matches_full_recomputation():
    dist = random_instance(15)
    tour = np.random.default_rng(1).permutation(15)
    base = tour_length(tour, dist)
    for i in range(15):
        for j in range(i + 1, 15 if i else 14):  # (0, n - 1) reverses the whole tour: no move
            moved = tour.copy()
            moved[i : j + 1] = moved[i : j + 1][::-1]
            assert two_opt_delta(tour, dist, i, j) == tour_length(moved, dist) - base


def test_two_opt_reaches_a_local_optimum():
    dist = random_instance(30, seed=2)
    start = np.random.default_rng(3).permutation(30)
    result = two_opt(start, dist)
    assert is_tour(result.tour, 30) and result.length < tour_length(start, dist)
    for i in range(30):
        for j in range(i + 1, 30):
            if not (i == 0 and j == 29):
                assert two_opt_delta(result.tour, dist, i, j) >= 0


def test_order_crossover_example():
    # the classic textbook example: segment 4 5 6 7 of p1 kept, the rest in p2's order after b
    p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) - 1
    p2 = np.array([9, 3, 7, 8, 2, 6, 5, 1, 4]) - 1
    child = order_crossover(p1, p2, 3, 7) + 1
    assert child.tolist() == [3, 8, 2, 4, 5, 6, 7, 1, 9]


@pytest.mark.parametrize("seed", range(20))
def test_order_crossover_always_gives_a_tour(seed):
    rng = np.random.default_rng(seed)
    p1, p2 = rng.permutation(10), rng.permutation(10)
    a, b = np.sort(rng.choice(11, 2, replace=False))
    child = order_crossover(p1, p2, a, b)
    assert is_tour(child, 10) and np.array_equal(child[a:b], p1[a:b])


def test_two_opt_mutation_reverses_a_segment_only():
    tour = np.arange(8)
    assert two_opt_mutation(tour, 2, 5).tolist() == [0, 1, 5, 4, 3, 2, 6, 7]
    assert tour.tolist() == list(range(8))  # the parent is not modified


def test_population_lengths():
    dist = random_instance(9)
    pop = np.array([np.random.default_rng(s).permutation(9) for s in range(5)])
    assert lengths(pop, dist).tolist() == [tour_length(t, dist) for t in pop]


@pytest.mark.parametrize("solver", [anneal, evolve])
def test_metaheuristics_find_the_optimum_of_small_instances(solver):
    dist = random_instance(8, seed=4)
    result = solver(dist, seed=0, evaluations=20_000)
    assert is_tour(result.tour, 8)
    assert result.length == tour_length(result.tour, dist) == brute_force_optimum(dist)


@pytest.mark.parametrize("solver", [anneal, evolve])
def test_metaheuristics_solve_the_circle(solver):
    result = solver(circle(), seed=1, evaluations=40_000)
    assert result.length == CIRCLE_OPT


@pytest.mark.parametrize("solver", [anneal, evolve])
def test_history_is_monotone_and_within_budget(solver):
    dist = random_instance(20)
    result = solver(dist, seed=0, evaluations=10_000)
    evals = [e for e, _ in result.history]
    best = [b for _, b in result.history]
    assert evals == sorted(evals) and evals[-1] <= 10_000 and result.evaluations <= 10_000
    assert best == sorted(best, reverse=True) and best[-1] == result.length


@pytest.mark.parametrize("solver", [anneal, evolve])
def test_same_seed_same_result(solver):
    dist = random_instance(20)
    a, b = solver(dist, seed=5, evaluations=5_000), solver(dist, seed=5, evaluations=5_000)
    assert np.array_equal(a.tour, b.tour) and a.history == b.history


def test_annealing_from_a_given_start_never_gets_worse():
    dist = random_instance(25)
    start = two_opt(nearest_neighbour(dist).tour, dist)
    result = anneal(dist, seed=0, evaluations=5_000, start=start.tour, t0=1e-9)
    assert result.length <= start.length
