"""The experiments behind results/: a parameter pilot and the main 10-seed comparison."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from . import ga, heuristics, sa
from .tsplib import DATA_DIR, INSTANCES, load, load_optimal_tour, tour_length

BUDGET = 1_000_000  # tour evaluations per GA or SA run
SEEDS = tuple(range(10))
PILOT_SEEDS = (100, 101, 102)  # disjoint from the reported seeds
PILOT_INSTANCES = ("st70", "kroA100")

# Chosen by the pilot (results/pilot.md); see choose() for the rule.
GA_PARAMS = {"population": 50, "mutation_rate": 0.6}
SA_PARAMS = {"accept": 0.5, "final_ratio": 1e-2}


def gap(length: float, optimum: int) -> float:
    """Excess over the optimum in percent."""
    return 100.0 * (length - optimum) / optimum


def run_sa(dist, seed, evaluations, accept=0.5, final_ratio=1e-3):
    rng = np.random.default_rng([seed, 1])  # start tour and temperature probe: own stream
    start = rng.permutation(len(dist))
    t0 = sa.initial_temperature(start.tolist(), dist.tolist(), rng, accept)
    return sa.anneal(dist, seed, evaluations, start=start, t0=t0, final_ratio=final_ratio)


def run_ga(dist, seed, evaluations, population=100, mutation_rate=0.3):
    return ga.evolve(dist, seed, evaluations, population=population, mutation_rate=mutation_rate)


def optimum_of(name: str, data_dir) -> tuple:
    """Instance, optimal tour and optimum (the length of TSPLIB's optimal tour file)."""
    inst = load(name, data_dir)
    tour = load_optimal_tour(name, data_dir)
    optimum = tour_length(tour, inst.dist)
    published = inst.optimum
    if published is not None and optimum != published:
        raise ValueError(f"{name}: optimal tour has length {optimum}, TSPLIB says {published}")
    return inst, tour, optimum


def pilot(
    evaluations: int = BUDGET,
    instances=PILOT_INSTANCES,
    seeds=PILOT_SEEDS,
    data_dir=DATA_DIR,
    log=print,
) -> dict:
    """Small grid over the main GA and SA parameters on two instances and three spare seeds."""
    rows = []
    grids = {
        "GA": [
            {"population": p, "mutation_rate": m}
            for p, m in itertools.product((50, 100, 200), (0.1, 0.3, 0.6))
        ],
        "SA": [
            {"accept": a, "final_ratio": f}
            for a, f in itertools.product((0.3, 0.5, 0.8), (1e-2, 1e-3, 1e-4))
        ],
    }
    runner = {"GA": run_ga, "SA": run_sa}
    for algorithm, grid in grids.items():
        for params in grid:
            gaps = {}
            for name in instances:
                inst, _, optimum = optimum_of(name, data_dir)
                run = runner[algorithm]
                found = [run(inst.dist, s, evaluations, **params).length for s in seeds]
                gaps[name] = gap(float(np.mean(found)), optimum)
            mean = float(np.mean(list(gaps.values())))
            rows.append({"algorithm": algorithm, "params": params, "gaps": gaps, "mean_gap": mean})
            log(f"pilot {algorithm} {params}: mean gap {mean:.2f} %  {gaps}")
    return {
        "evaluations": evaluations,
        "seeds": list(seeds),
        "instances": list(instances),
        "rows": rows,
    }


def choose(pilot_results: dict) -> dict[str, dict]:
    """The parameters with the lowest mean gap over the pilot instances, per algorithm."""
    best = {}
    for row in pilot_results["rows"]:
        current = best.get(row["algorithm"])
        if current is None or row["mean_gap"] < current["mean_gap"]:
            best[row["algorithm"]] = row
    return {algorithm: row["params"] for algorithm, row in best.items()}


def summarise(lengths: list[int], seconds: list[float], optimum: int) -> dict:
    arr = np.array(lengths)
    return {
        "lengths": lengths,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "best": int(arr.min()),
        "worst": int(arr.max()),
        "mean_gap": gap(arr.mean(), optimum),
        "best_gap": gap(arr.min(), optimum),
        "optimal_runs": int(np.sum(arr == optimum)),
        "mean_seconds": float(np.mean(seconds)),
    }


def main_experiment(
    instances=INSTANCES,
    seeds=SEEDS,
    evaluations: int = BUDGET,
    ga_params: dict | None = None,
    sa_params: dict | None = None,
    data_dir=DATA_DIR,
    log=print,
) -> dict:
    """NN, NN + 2-opt, GA and SA on every instance with every seed.

    NN starts from a random city per seed (it is deterministic otherwise); NN + 2-opt improves that
    tour by best-improvement 2-opt. GA and SA start from random tours and get the same budget.
    """
    ga_params = GA_PARAMS if ga_params is None else ga_params
    sa_params = SA_PARAMS if sa_params is None else sa_params
    results = {
        "evaluations": evaluations,
        "seeds": list(seeds),
        "ga_params": ga_params,
        "sa_params": sa_params,
        "instances": {},
    }
    for name in instances:
        inst, optimal_tour, optimum = optimum_of(name, data_dir)
        runs = {"NN": [], "NN+2opt": [], "GA": [], "SA": []}
        for seed in seeds:
            start = int(np.random.default_rng(seed).integers(inst.n))
            nn = heuristics.nearest_neighbour(inst.dist, start)
            runs["NN"].append(nn)
            runs["NN+2opt"].append(heuristics.two_opt(nn.tour, inst.dist))
            runs["GA"].append(run_ga(inst.dist, seed, evaluations, **ga_params))
            runs["SA"].append(run_sa(inst.dist, seed, evaluations, **sa_params))
            log(
                f"{name:8s} seed {seed}: "
                + "  ".join(
                    f"{a} {r[-1].length} ({gap(r[-1].length, optimum):.1f} %)"
                    for a, r in runs.items()
                )
                + f"  [GA {runs['GA'][-1].seconds:.1f} s, SA {runs['SA'][-1].seconds:.1f} s]"
            )
        entry = {"n": inst.n, "optimum": optimum, "comment": inst.comment, "algorithms": {}}
        for algorithm, rs in runs.items():
            summary = summarise([r.length for r in rs], [r.seconds for r in rs], optimum)
            best = min(rs, key=lambda r: r.length)
            summary["best_tour"] = best.tour.tolist()
            if algorithm in ("GA", "SA"):
                summary["histories"] = [r.history for r in rs]
            entry["algorithms"][algorithm] = summary
        entry["coords"] = inst.coords.tolist()
        entry["optimal_tour"] = optimal_tour.tolist()
        results["instances"][name] = entry
    return results


HEAVY = ("best_tour", "histories", "coords", "optimal_tour", "lengths")


def write_json(data: dict, out: Path) -> None:
    """``results.json``: readable summary. ``runs.json``: everything, incl. tours and histories."""
    (out / "runs.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    light = json.loads(json.dumps(data))
    for entry in light["instances"].values():
        for key in HEAVY:
            entry.pop(key, None)
        for summary in entry["algorithms"].values():
            summary["lengths"] = " ".join(map(str, summary["lengths"]))
            for key in HEAVY[:2]:
                summary.pop(key, None)
    (out / "results.json").write_text(json.dumps(light, indent=2) + "\n")
