"""Tests for the mibiremo.wells module."""

import math
import pandas as pd
import pytest
from mibiremo.wells import Well
from mibiremo.wells import read_wells
from mibiremo.wells import array_linear
from mibiremo.wells import array_radial
from mibiremo.wells import write_wells

# Example injection well template (field location in EPSG:27572)
GEOMETRY = {"well_top": 96.16, "well_bottom": 90.5, "diameter": 0.1, "screen_top": 95.5, "screen_bottom": 90.5}
INJ = Well("INJ", x=222049.6162, y=2401812.2549, **GEOMETRY)


def test_screen_within_well():
    with pytest.raises(ValueError, match="within the well"):
        Well("W", x=0.0, y=0.0, **(GEOMETRY | {"screen_bottom": 90.0}))


def test_array_radial_clockwise():
    """Three copies of an injection well on a 4 m circle, the first one starting north."""
    template = Well("INJ", x=10.0, y=20.0, **GEOMETRY)
    wells = array_radial(template, 3, radius=4.0)
    s = 4.0 * math.sqrt(3) / 2
    expected = [(10.0, 24.0), (10.0 + s, 18.0), (10.0 - s, 18.0)]  # azimuths 0°, 120°, 240°
    assert [(w.x, w.y) for w in wells] == [pytest.approx(p) for p in expected]
    assert [w.name for w in wells] == ["INJ_1", "INJ_2", "INJ_3"]
    assert all(w.screen_top == template.screen_top and w.diameter == template.diameter for w in wells)


def test_array_linear():
    """Test array perpendicular to groundwater flow."""
    flow_azimuth = math.radians(22.0)  # Groundwater flow direction
    template = Well("INJ", x=0.0, y=0.0, **GEOMETRY)
    wells = array_linear(template, 4, spacing=5.0, azimuth=22.0 + 90.0)
    along_flow = [w.x * math.sin(flow_azimuth) + w.y * math.cos(flow_azimuth) for w in wells]
    across_flow = [w.x * math.cos(flow_azimuth) - w.y * math.sin(flow_azimuth) for w in wells]
    assert along_flow == pytest.approx([0.0] * 4, abs=1e-12)
    assert across_flow == pytest.approx([-7.5, -2.5, 2.5, 7.5])


@pytest.mark.parametrize("suffix", [".shp", ".csv"])
def test_write_read(tmp_path, suffix):
    extraction = Well("EXT_1", x=222049.7016, y=2401808.1922, **(GEOMETRY | {"well_top": 96.35}))
    path = tmp_path / f"wells{suffix}"
    write_wells([extraction, INJ], path, crs="EPSG:27572")
    assert read_wells(path) == [extraction, INJ]


def test_geographic_coordinates(tmp_path):
    """Wells in degrees (geographic CRS) are rejected: the model uses x and y in metres."""
    path = tmp_path / "wells.gpkg"
    write_wells([Well("INJ", x=2.35, y=48.85, **GEOMETRY)], path, crs="EPSG:4326")
    with pytest.raises(ValueError, match="geographic"):
        read_wells(path)


def test_read_locations(tmp_path):
    path = tmp_path / "wells.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).to_csv(path, index=False)
    wells = read_wells(path, well=INJ)
    assert [(w.name, w.x, w.y) for w in wells] == [("INJ_1", 1.0, 3.0), ("INJ_2", 2.0, 4.0)]
    assert all(w.diameter == INJ.diameter and w.screen_top == INJ.screen_top for w in wells)
