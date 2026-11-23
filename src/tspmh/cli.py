"""Command line: ``tspmh solve``, ``tspmh pilot`` and ``tspmh experiment``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__, experiment, heuristics
from .tsplib import DATA_DIR, INSTANCES, load, parse_tsp

ALGORITHMS = ("nn", "2opt", "ga", "sa")


def load_any(spec: str, data_dir: str):
    """An instance by TSPLIB name (from ``data_dir``) or by the path of a ``.tsp`` file."""
    path = Path(spec)
    if path.suffix == ".tsp" and path.exists():
        return parse_tsp(path.read_text())
    return load(spec, data_dir)


def cmd_solve(args) -> int:
    inst = load_any(args.instance, args.data)
    if args.algorithm in ("nn", "2opt"):
        result = heuristics.nearest_neighbour(inst.dist, args.seed % inst.n)
        if args.algorithm == "2opt":
            result = heuristics.two_opt(result.tour, inst.dist)
    elif args.algorithm == "ga":
        result = experiment.run_ga(inst.dist, args.seed, args.evaluations, **experiment.GA_PARAMS)
    else:
        result = experiment.run_sa(inst.dist, args.seed, args.evaluations, **experiment.SA_PARAMS)
    line = f"{inst.name}: {result.algorithm} length {result.length:,}"
    if inst.optimum:
        gap = experiment.gap(result.length, inst.optimum)
        line += f" (optimum {inst.optimum:,}, gap {gap:.2f} %)"
    print(f"{line} in {result.seconds:.2f} s")
    if args.tour:
        print(" ".join(str(c + 1) for c in result.tour))  # TSPLIB numbering
    return 0


def cmd_pilot(args) -> int:
    from . import report

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = experiment.pilot(args.evaluations)
    chosen = experiment.choose(results)
    (out / "pilot.json").write_text(json.dumps({**results, "chosen": chosen}, indent=2) + "\n")
    (out / "pilot.md").write_text(report.pilot_markdown(results, chosen))
    print(f"chosen: {chosen}; wrote {out}/pilot.md and pilot.json")
    if chosen != {"GA": experiment.GA_PARAMS, "SA": experiment.SA_PARAMS}:
        print("note: differs from GA_PARAMS / SA_PARAMS in experiment.py; update them")
    return 0


def cmd_experiment(args) -> int:
    from . import plots, report

    start = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = experiment.main_experiment(
        instances=args.instances, seeds=range(args.seeds), evaluations=args.evaluations
    )
    experiment.write_json(results, out)
    (out / "RESULTS.md").write_text(report.markdown(results))
    plots.convergence(results, out / "convergence.png")
    plots.tours(results, out / "tours.png")
    print(f"wrote {out}/RESULTS.md, results.json, runs.json and two figures "
          f"({time.perf_counter() - start:.0f} s)")  # fmt: skip
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tspmh", description=__doc__)
    p.add_argument("--version", action="version", version=f"tspmh {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("solve", help="run one algorithm on one instance")
    s.add_argument("instance", help=f"TSPLIB name ({', '.join(INSTANCES)}) or a .tsp file")
    s.add_argument("-a", "--algorithm", choices=ALGORITHMS, default="sa")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--evaluations", type=int, default=experiment.BUDGET)
    s.add_argument("--tour", action="store_true", help="also print the tour (cities from 1)")
    s.add_argument("--data", default=str(DATA_DIR), help="directory with the .tsp files")
    s.set_defaults(func=cmd_solve)

    pl = sub.add_parser("pilot", help="parameter pilot on spare seeds -> results/pilot.md")
    pl.add_argument("--evaluations", type=int, default=experiment.BUDGET)
    pl.add_argument("--out", default="results")
    pl.set_defaults(func=cmd_pilot)

    e = sub.add_parser("experiment", help="all algorithms, instances and seeds -> results/")
    e.add_argument("--instances", nargs="+", default=list(INSTANCES), choices=INSTANCES)
    e.add_argument("--seeds", type=int, default=len(experiment.SEEDS), help="seeds 0..N-1")
    e.add_argument("--evaluations", type=int, default=experiment.BUDGET)
    e.add_argument("--out", default="results")
    e.set_defaults(func=cmd_experiment)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
