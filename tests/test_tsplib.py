from pathlib import Path

import numpy as np
import pytest

from tspmh.tsplib import (
    DATA_DIR,
    INSTANCES,
    distance_matrix,
    load,
    load_optimal_tour,
    parse_tour,
    parse_tsp,
    tour_length,
)

SQUARE = """NAME : square
COMMENT : 4 corners of a 3 x 4 rectangle
TYPE : TSP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 3 0
3 3 4
4 0 4
EOF
"""


def test_parse_tsp():
    inst = parse_tsp(SQUARE)
    assert inst.name == "square" and inst.n == 4 and inst.comment.startswith("4 corners")
    assert inst.dist[0, 2] == 5 and inst.dist[0, 1] == 3
    assert tour_length([0, 1, 2, 3], inst.dist) == 14
    assert tour_length([0, 2, 1, 3], inst.dist) == 18
    assert inst.optimum is None


def test_header_without_spaces_around_colon():
    inst = parse_tsp(SQUARE.replace("NAME : square", "NAME: square").replace(" : ", ": "))
    assert inst.name == "square" and inst.n == 4


def test_euc_2d_rounds_half_up():
    coords = np.array([[0.0, 0.0], [1.5, 2.0], [0.5, 0.0], [1.0, 1.0]])
    d = distance_matrix(coords)
    assert d[0, 1] == 3  # 2.5 -> 3
    assert d[0, 2] == 1  # 0.5 -> 1
    assert d[0, 3] == 1  # 1.414 -> 1
    assert distance_matrix(coords, "CEIL_2D")[0, 3] == 2


def test_att_distance_rounds_up_when_nint_falls_short():
    # r = sqrt((10^2 + 0^2) / 10) = 3.162, nint 3 < r -> 4; r = sqrt(250 / 10) = 5 exactly -> 5
    d = distance_matrix(np.array([[0.0, 0.0], [10.0, 0.0], [15.0, 5.0]]), "ATT")
    assert d[0, 1] == 4 and d[0, 2] == 5


def test_bad_input():
    with pytest.raises(ValueError, match="DIMENSION is 4 but 3"):
        parse_tsp(SQUARE.replace("4 0 4\n", ""))
    with pytest.raises(ValueError, match="unsupported"):
        parse_tsp(SQUARE.replace("EUC_2D", "GEO"))
    with pytest.raises(ValueError, match="not a symmetric TSP"):
        parse_tsp(SQUARE.replace("TYPE : TSP", "TYPE : ATSP"))


def test_parse_tour():
    text = "NAME : t\nTYPE : TOUR\nDIMENSION : 4\nTOUR_SECTION\n1\n3\n2\n4\n-1\nEOF\n"
    assert parse_tour(text).tolist() == [0, 2, 1, 3]
    with pytest.raises(ValueError, match="exactly once"):
        parse_tour(text.replace("4\n-1", "3\n-1"))


def test_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load("berlin52", tmp_path)


@pytest.mark.skipif(not (Path(DATA_DIR) / "kroA100.tsp").exists(), reason="run make data first")
@pytest.mark.parametrize("name", INSTANCES)
def test_published_optimal_tours_have_the_published_length(name):
    """The strongest check of the distance function: TSPLIB's optimal tours give its optima."""
    inst = load(name)
    assert tour_length(load_optimal_tour(name), inst.dist) == inst.optimum
