"""Tests for the mibiremo.plotting module."""

import matplotlib.pyplot as plt
import numpy as np
from mibiremo import plotting
from mibiremo.field import FieldModel
from mibiremo.field import intermittent_pumping
from mibiremo.wells import Well

GEOMETRY = {"well_top": 96.3, "well_bottom": 90.5, "diameter": 0.1, "screen_top": 95.5, "screen_bottom": 92.0}
WELLS = [
    Well("EXT_1", x=0.0, y=0.0, **GEOMETRY),
    Well("INJ_1", x=0.0, y=4.0, **GEOMETRY),  # north of EXT_1
    Well("MON_1", x=4.0, y=0.0, **GEOMETRY),  # east of EXT_1
]
FLOW_RATES = {"EXT_1": -3e-5, "INJ_1": intermittent_pumping(1e-5, 3600.0, 3600.0, end_time=86400.0)}


def test_well_symbols():
    """Q < 0: extraction (circle); Q > 0 at any time: injection (triangle); no Q: monitoring (square).

    The flow arrow follows the azimuth convention, clockwise from north: 90° points east.
    """
    ax = plt.subplots()[1]
    plotting.wells(ax, WELLS, FLOW_RATES, labels=False)
    assert [line.get_marker() for line in ax.lines] == ["o", "^", "s"]

    plotting.flow_arrow(ax, azimuth=90.0)
    arrow = ax.texts[0]
    dx, dy = np.subtract(arrow.xy, arrow.xyann)
    assert dx > 0
    assert abs(dy) < 1e-9 * dx


def _model(workspace, **kwargs):
    """Built single-layer model of the three wells."""
    model = FieldModel(
        workspace=workspace,
        wells=WELLS,
        flow_rates=FLOW_RATES,
        domain_size=(40.0, 40.0),
        top=95.5,
        layer_bottom=90.5,
        hydraulic_conductivity=5e-6,
        porosity=0.25,
        reference_head=93.7,
        grid_spacing=4.0,
        grid_spacing_at_wells=1.0,
        simulation_time=86400.0,
        time_step=86400.0,
        **kwargs,
    )
    model.build()
    return model


def test_grid(tmp_path):
    """Check that the grid plot marks the cells of the lateral boundary.

    For a model with a general-head boundary (GHB), the coloured cells are all the cells on the perimeter of the grid
    and none inside it. The setting `boundary=False` disables the legend.
    """
    model = _model(tmp_path, boundary_conductance=1e-4)
    ax = plotting.grid(model)
    coloured = ~np.ma.getmaskarray(ax.collections[-1].get_array())  # (row, column), True where a cell is coloured
    interior = coloured[1:-1, 1:-1]
    assert not interior.any()
    assert coloured.sum() == coloured.size - interior.size
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["general-head boundary (GHB)"]

    assert plotting.grid(model, boundary=False).get_legend() is None


def test_cross_section(tmp_path):
    """Check that a cross section shows the screens of the wells located on its line.

    A south-north section through EXT_1 draws the screens of EXT_1 and INJ_1, from the screen bottom
    to the screen top. MON_1, 4 m east of the line, is not drawn. The map view is only drawn, not checked.
    """
    model = _model(tmp_path)
    values = np.zeros(model.grid.shape)
    plotting.map_view(model, values)

    ax = plotting.cross_section(model, values, line=[(0.0, -10.0), (0.0, 10.0)])
    screens = {text.get_text(): line.get_xydata().tolist() for text, line in zip(ax.texts, ax.lines)}
    # MON_1 should not appear in the cross section.
    assert screens == {"EXT_1": [[0.0, 92.0], [0.0, 95.5]], "INJ_1": [[4.0, 92.0], [4.0, 95.5]]}
