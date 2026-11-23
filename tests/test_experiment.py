import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from tspmh import experiment, plots, report
from tspmh.cli import main
from tspmh.tsplib import distance_matrix, tour_length

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def toy_dir(tmp_path):
    """Two random 8-city instances in TSPLIB format, with brute-forced optimal tours."""
    rng = np.random.default_rng(0)
    for name in ("toyA", "toyB"):
        coords = rng.integers(0, 100, (8, 2))
        lines = [f"NAME : {name}", "COMMENT : random toy", "TYPE : TSP", "DIMENSION : 8",
                 "EDGE_WEIGHT_TYPE : EUC_2D", "NODE_COORD_SECTION"]  # fmt: skip
        lines += [f"{i + 1} {x} {y}" for i, (x, y) in enumerate(coords)] + ["EOF"]
        (tmp_path / f"{name}.tsp").write_text("\n".join(lines) + "\n")
        dist = distance_matrix(coords.astype(float))
        best = min(([0, *p] for p in itertools.permutations(range(1, 8))),
                   key=lambda t: tour_length(t, dist))  # fmt: skip
        tour_text = "\n".join(str(c + 1) for c in best)
        (tmp_path / f"{name}.opt.tour").write_text(f"TOUR_SECTION\n{tour_text}\n-1\nEOF\n")
    return tmp_path


def test_gap():
    assert experiment.gap(110, 100) == pytest.approx(10.0)


def test_main_experiment_and_outputs(toy_dir, tmp_path):
    results = experiment.main_experiment(
        instances=("toyA", "toyB"),
        seeds=range(3),
        evaluations=3_000,
        data_dir=toy_dir,
        log=lambda *_: None,
    )
    a = results["instances"]["toyA"]["algorithms"]
    assert set(a) == {"NN", "NN+2opt", "GA", "SA"}
    assert len(a["SA"]["histories"]) == 3 and a["NN+2opt"]["mean"] <= a["NN"]["mean"]
    assert all(s["best_gap"] >= 0 for s in a.values())  # nothing beats the brute-force optimum
    assert a["SA"]["best_gap"] == 0 and a["SA"]["optimal_runs"] >= 1

    out = tmp_path / "out"
    out.mkdir()
    experiment.write_json(results, out)
    light = json.loads((out / "results.json").read_text())
    assert "histories" not in light["instances"]["toyA"]["algorithms"]["GA"]
    assert "coords" not in light["instances"]["toyA"]
    assert json.loads((out / "runs.json").read_text())["instances"]["toyA"]["coords"]

    md = report.markdown(results)
    assert "| toyA | 8 |" in md and "Optimal runs" in md
    plots.convergence(results, out / "c.png")
    plots.tours(results, out / "t.png")
    assert (out / "c.png").stat().st_size > 0 and (out / "t.png").stat().st_size > 0


def test_pilot_and_choose(toy_dir):
    results = experiment.pilot(
        evaluations=2_000, instances=("toyA",), seeds=(0,), data_dir=toy_dir, log=lambda *_: None
    )
    assert len(results["rows"]) == 18
    chosen = experiment.choose(results)
    assert set(chosen) == {"GA", "SA"}
    assert "**(chosen)**" in report.pilot_markdown(results, chosen)


def test_choose_picks_the_lowest_mean_gap():
    rows = [
        {"algorithm": "GA", "params": {"population": 50}, "mean_gap": 3.0},
        {"algorithm": "GA", "params": {"population": 100}, "mean_gap": 2.0},
        {"algorithm": "SA", "params": {"accept": 0.5}, "mean_gap": 1.0},
    ]
    assert experiment.choose({"rows": rows}) == {"GA": {"population": 100}, "SA": {"accept": 0.5}}


@pytest.mark.skipif(not (ROOT / "results" / "pilot.json").exists(), reason="no pilot results")
def test_parameters_in_code_match_the_committed_pilot():
    pilot = json.loads((ROOT / "results" / "pilot.json").read_text())
    assert experiment.choose(pilot) == {"GA": experiment.GA_PARAMS, "SA": experiment.SA_PARAMS}


def test_optimum_must_match_tsplib(toy_dir):
    (toy_dir / "eil51.tsp").write_text((toy_dir / "toyA.tsp").read_text().replace("toyA", "eil51"))
    (toy_dir / "eil51.opt.tour").write_text((toy_dir / "toyA.opt.tour").read_text())
    with pytest.raises(ValueError, match="TSPLIB says 426"):
        experiment.optimum_of("eil51", toy_dir)


def test_cli_solve_on_a_tsp_file(toy_dir, capsys):
    path = toy_dir / "toyA.tsp"
    for algorithm in ("nn", "2opt", "ga", "sa"):
        args = ["solve", str(path), "-a", algorithm, "--evaluations", "2000", "--tour"]
        assert main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("toyA: NN length")
    assert sorted(map(int, lines[1].split())) == list(range(1, 9))


def test_cli_missing_instance(tmp_path, capsys):
    assert main(["solve", "berlin52", "--data", str(tmp_path)]) == 2
    assert "download_data" in capsys.readouterr().err
