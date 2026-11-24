"""Code parameters [[n, k, d]] copied from the literature, each with its source and location.

These are NOT computed here. Every row was read from the full text of a paper ingested into
chatQEC's knowledge base (arXiv sources, verified by title and authors), and
tests/test_paper_codes.py re-checks each row against that text. ``locator`` names the table,
section, or construction (check weight, layout) so an answer can cite it.

Families:
  bb5               bivariate-bicycle codes with weight-5 checks
  tile              tile codes, local on a planar 2D lattice with open boundaries
  directional_tile  planar tile codes with nearest-neighbour-gate syndrome extraction
  barbell           qLDPC codes for IQM's six-qubit star lattice with non-local couplers

Not included: surface-code and BB6 baselines that the papers print only for comparison, the GB4,
toric and concatenated codes of the hardware demonstration, and simulation labels whose code
is not tied to a stated family.
"""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES = ("bb5", "tile", "directional_tile", "barbell")


@dataclass(frozen=True)
class PaperCode:
    family: str
    n: int
    k: int
    d: int
    source: str
    locator: str


def _group(
    family: str, source: str, locator: str, *triples: tuple[int, int, int]
) -> tuple[PaperCode, ...]:
    return tuple(PaperCode(family, n, k, d, source, locator) for n, k, d in triples)


_BB5_SOURCE = "arXiv:2503.22071"  # Ye and Delfosse, "Quantum error correction for long chains of trapped ions"
_BB5_DEMO = "arXiv:2606.06455"  # Breakeven demonstration of quantum LDPC codes (trapped ions)
_TILE = "arXiv:2504.09171"  # Tile Codes: High-Efficiency Quantum Codes on a Lattice with Boundary
_DIRECTIONAL = "arXiv:2606.19482"  # Nearest-neighbour gates are all you need
_BARBELL = "arXiv:2606.06062"  # Barbell Codes: qLDPC Codes for Superconducting Quantum Hardware
_BARBELL_TABLE = "Table tab:barbellcodetiles (six-qubit star lattice + non-local couplers)"

_PAPER_CODES: tuple[PaperCode, ...] = (
    *_group(
        "bb5",
        _BB5_SOURCE,
        "Table tab:baseline_code_parameters: BB5 codes, stabilizer weight 5",
        (30, 4, 5),
        (48, 4, 7),
    ),
    *_group(
        "bb5",
        _BB5_DEMO,
        "Appendix: three BB5 instances with parameters [[6d,4,d]], d in {3,4,5}",
        (18, 4, 3),
        (24, 4, 4),
        (30, 4, 5),
    ),
    *_group(
        "tile",
        _TILE,
        "Abstract: 3x3 box, weight-6 stabilizers",
        (288, 8, 12),
    ),
    *_group(
        "tile",
        _TILE,
        "Abstract: weight-8 stabilizers",
        (288, 8, 14),
    ),
    *_group(
        "tile",
        _TILE,
        "Introduction: 4x4 box, weight-8 stabilizers",
        (288, 18, 13),
    ),
    *_group(
        "tile",
        _TILE,
        "Abstract: weight-8 stabilizers, slightly higher non-locality",
        (512, 18, 19),
    ),
    *_group(
        "directional_tile",
        _DIRECTIONAL,
        "Table tab:code_para: word N^2ESEN^2, weight 7, square grid",
        (60, 4, 5),
        (180, 4, 9),
    ),
    *_group(
        "directional_tile",
        _DIRECTIONAL,
        "Table tab:code_para: word N^2E^2SE^2N^2, weight 9, square grid",
        (162, 10, 6),
        (351, 10, 9),
    ),
    *_group(
        "directional_tile",
        _DIRECTIONAL,
        "Table tab:code_para: word N^2E^2SESE^2N^2, weight 11, square grid",
        (241, 14, 12),
        (323, 14, 15),
    ),
    *_group(
        "directional_tile",
        _DIRECTIONAL,
        "Table tab:code_para: word ENES^2WS^2ENE, weight 11, hexagonal grid",
        (52, 10, 3),
        (199, 10, 7),
    ),
    *_group(
        "directional_tile",
        _DIRECTIONAL,
        "Appendix table of the N^2ESEN^2 family (weight 7), k = 4",
        (24, 4, 3),
        (40, 4, 4),
        (84, 4, 6),
        (112, 4, 7),
        (144, 4, 8),
    ),
    *_group(
        "barbell",
        _BARBELL,
        f"{_BARBELL_TABLE}: tile row 1 of 4 (D=2, v_XZ=(0,0)), k = 7",
        (128, 7, 6),
        (162, 7, 7),
        (200, 7, 8),
        (242, 7, 9),
        (288, 7, 11),
        (338, 7, 13),
        (392, 7, 14),
    ),
    *_group(
        "barbell",
        _BARBELL,
        f"{_BARBELL_TABLE}: tile row 2 of 4 (D=2, v_XZ=(-1,0)), k = 8",
        (128, 8, 6),
        (162, 8, 7),
        (200, 8, 9),
        (242, 8, 10),
        (288, 8, 12),
        (338, 8, 13),
        (392, 8, 15),
    ),
    *_group(
        "barbell",
        _BARBELL,
        f"{_BARBELL_TABLE}: tile row 3 of 4 (D=3, v_XZ=(0,0)), k = 16",
        (242, 16, 8),
        (288, 16, 9),
        (338, 16, 10),
        (392, 16, 11),
        (450, 16, 14),
    ),
    *_group(
        "barbell",
        _BARBELL,
        f"{_BARBELL_TABLE}: tile row 4 of 4 (D=3, v_XZ=(0,0)), k = 16",
        (162, 16, 8),
        (200, 16, 10),
        (242, 16, 11),
        (288, 16, 12),
        (338, 16, 13),
        (392, 16, 14),
        (450, 16, 15),
        (512, 16, 16),
    ),
)


def paper_codes(family: str) -> list[PaperCode]:
    """Every row of ``family``, in the order the tables list them.

    Raises:
        KeyError: ``family`` is not one of FAMILIES.
    """
    if family not in FAMILIES:
        raise KeyError(f"unknown literature code family: {family!r}")
    return [row for row in _PAPER_CODES if row.family == family]
