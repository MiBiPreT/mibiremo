"""Tests for the mibiremo.field module."""

from dataclasses import replace
import numpy as np
import pytest
from mibiremo.field import structured_grid
from mibiremo.wells import Well
from mibiremo.wells import array_radial

# Example field locations (EPSG:27572): extraction well EXT_1, injection wells INJ_1-3 about 4 m around it
GEOMETRY = {"well_top": 96.3, "well_bottom": 90.5, "diameter": 0.1, "screen_top": 95.5, "screen_bottom": 90.5}
EXTRACTION = Well("EXT_1", x=222049.7016, y=2401808.1922, **GEOMETRY)
DESIGN = [EXTRACTION, *array_radial(replace(EXTRACTION, name="INJ"), n_wells=3, radius=4.0)]
REAL = [
    EXTRACTION,
    Well("INJ_1", x=222049.6162, y=2401812.2549, **GEOMETRY),  # 0.085 m west of EXT_1
    Well("INJ_2", x=222046.1198, y=2401806.2488, **GEOMETRY),
    Well("INJ_3", x=222053.1180, y=2401806.2389, **GEOMETRY),  # 0.010 m south of INJ_2
]
GRID = {"domain_size": (148.0, 116.0), "top": 95.5, "bottom": 90.5, "n_layers": 7, "grid_spacing": 4.0}
FINE = 0.5  # grid spacing at the wells [m]


@pytest.mark.parametrize(
    ("wells", "tolerance"),
    [(DESIGN, 1e-6), (REAL, FINE / 4)],  # real wells nearly in line share a column (row)
)
def test_wells_at_cell_centres(wells, tolerance):
    """Wells are at cell centres of a domain centred on them, with layers of equal thickness."""
    grid = structured_grid(wells, grid_spacing_at_wells=FINE, **GRID)
    for w in wells:
        row, column = grid.intersect(w.x, w.y)
        assert grid.xcellcenters[row, column] == pytest.approx(w.x, abs=tolerance)
        assert grid.ycellcenters[row, column] == pytest.approx(w.y, abs=tolerance)
    x_min, x_max, y_min, y_max = grid.extent
    x, y = [w.x for w in wells], [w.y for w in wells]
    assert (x_max - x_min, y_max - y_min) == pytest.approx(GRID["domain_size"])
    assert ((x_min + x_max) / 2, (y_min + y_max) / 2) == pytest.approx(((min(x) + max(x)) / 2, (min(y) + max(y)) / 2))
    assert np.diff(grid.botm[:, 0, 0]) == pytest.approx([-5.0 / 7] * 6)  # layers of equal thickness


def test_cell_widths():
    """Cell widths are fine near the wells and grow by at most 1.5 up to the grid spacing."""
    margin = 5.0
    grid = structured_grid(DESIGN, grid_spacing_at_wells=FINE, refinement_margin=margin, **GRID)
    for widths, centres, wells_xy in [
        (grid.delr, grid.xcellcenters[0], [w.x for w in DESIGN]),
        (grid.delc, grid.ycellcenters[:, 0], [w.y for w in DESIGN]),
    ]:
        ratio = widths[1:] / widths[:-1]
        assert np.maximum(ratio, 1 / ratio).max() <= 1.5 + 1e-9
        assert widths.max() <= GRID["grid_spacing"] + 1e-9
        near_wells = (centres > min(wells_xy) - margin) & (centres < max(wells_xy) + margin)
        assert widths[near_wells] == pytest.approx(FINE, rel=0.05)
