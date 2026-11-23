"""Reading TSPLIB instances (``.tsp``) and tours (``.tour``), with TSPLIB's integer distances.

TSPLIB defines every distance as an integer, and the published optima are only reproducible with
exactly its rounding rules (TSPLIB 95 documentation, section 2). For ``EUC_2D`` the distance is the
Euclidean distance rounded to the nearest integer, ``nint(sqrt(dx^2 + dy^2))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path("data") / "tsplib"

# Optimal tour lengths published in TSPLIB (TSP-BEST.html), all proven optimal.
OPTIMA = {"berlin52": 7542, "eil51": 426, "st70": 675, "kroA100": 21282}
INSTANCES = tuple(OPTIMA)


@dataclass
class Instance:
    name: str
    coords: np.ndarray  # (n, 2)
    edge_weight_type: str = "EUC_2D"
    comment: str = ""
    dist: np.ndarray = field(init=False, repr=False)  # (n, n) int64

    def __post_init__(self) -> None:
        self.dist = distance_matrix(self.coords, self.edge_weight_type)

    @property
    def n(self) -> int:
        return len(self.coords)

    @property
    def optimum(self) -> int | None:
        return OPTIMA.get(self.name)


def nint(x: np.ndarray) -> np.ndarray:
    """TSPLIB's nint: round half up, ``(int)(x + 0.5)`` for the non-negative values used here."""
    return np.floor(x + 0.5).astype(np.int64)


def distance_matrix(coords: np.ndarray, kind: str = "EUC_2D") -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1))
    if kind == "EUC_2D":
        return nint(d)
    if kind == "CEIL_2D":
        return np.ceil(d).astype(np.int64)
    if kind == "ATT":  # pseudo-Euclidean distance of att48 and att532
        r = np.sqrt((diff**2).sum(axis=-1) / 10.0)
        t = nint(r)
        return np.where(t < r, t + 1, t)
    raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {kind!r} (EUC_2D, CEIL_2D or ATT)")


def _header(lines: list[str]) -> tuple[dict[str, str], int]:
    """Parse ``KEY : value`` lines up to the first section keyword; return the index after it."""
    header = {}
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.endswith("_SECTION"):
            header["SECTION"] = line
            return header, i + 1
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"line {i + 1}: expected 'KEY : value', got {line!r}")
        header[key.strip()] = value.strip()
    raise ValueError("no data section found")


def parse_tsp(text: str) -> Instance:
    lines = text.splitlines()
    header, start = _header(lines)
    if header.get("TYPE", "TSP") != "TSP":
        raise ValueError(f"not a symmetric TSP: TYPE = {header['TYPE']}")
    if header["SECTION"] != "NODE_COORD_SECTION":
        raise ValueError(f"only NODE_COORD_SECTION is supported, got {header['SECTION']}")
    n = int(header["DIMENSION"])
    coords = np.zeros((n, 2))
    seen = 0
    for line in lines[start:]:
        parts = line.split()
        if not parts or parts[0] == "EOF":
            break
        node = int(parts[0])
        if not 1 <= node <= n:
            raise ValueError(f"node {node} outside 1..{n}")
        coords[node - 1] = float(parts[1]), float(parts[2])
        seen += 1
    if seen != n:
        raise ValueError(f"DIMENSION is {n} but {seen} coordinates were read")
    return Instance(
        name=header.get("NAME", "unnamed"),
        coords=coords,
        edge_weight_type=header.get("EDGE_WEIGHT_TYPE", "EUC_2D"),
        comment=header.get("COMMENT", ""),
    )


def parse_tour(text: str) -> np.ndarray:
    """A TSPLIB ``.tour`` file as a 0-based array of cities."""
    lines = text.splitlines()
    header, start = _header(lines)
    tour = []
    for line in lines[start:]:
        for token in line.split():
            if token in ("-1", "EOF"):
                return check_tour(np.array(tour) - 1, int(header.get("DIMENSION", len(tour))))
            tour.append(int(token))
    return check_tour(np.array(tour) - 1, int(header.get("DIMENSION", len(tour))))


def check_tour(tour: np.ndarray, n: int) -> np.ndarray:
    if len(tour) != n or not np.array_equal(np.sort(tour), np.arange(n)):
        raise ValueError("a tour must visit every city exactly once")
    return tour


def load(name: str, directory: str | Path = DATA_DIR) -> Instance:
    path = Path(directory) / f"{name}.tsp"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/download_data.sh first")
    return parse_tsp(path.read_text())


def load_optimal_tour(name: str, directory: str | Path = DATA_DIR) -> np.ndarray:
    return parse_tour((Path(directory) / f"{name}.opt.tour").read_text())


def tour_length(tour, dist: np.ndarray) -> int:
    tour = np.asarray(tour)
    return int(dist[tour, np.roll(tour, -1)].sum())
