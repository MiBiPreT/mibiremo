"""Wells (injection, extraction, or monitoring): geometry, layouts, and file import and export."""

import math
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
import pandas as pd


@dataclass
class Well:
    """Geometry of a well with a vertical screen.

    Azimuths in this module are measured in degrees clockwise from north (the y axis).

    Args:
        name: Well name.
        x: x coordinate (easting) [m].
        y: y coordinate (northing) [m].
        well_top: Elevation of the top of the well [m].
        well_bottom: Elevation of the bottom of the well [m].
        diameter: Well diameter [m].
        screen_top: Elevation of the top of the screen [m].
        screen_bottom: Elevation of the bottom of the screen [m].
    """

    name: str
    x: float
    y: float
    well_top: float
    well_bottom: float
    diameter: float
    screen_top: float
    screen_bottom: float

    def __post_init__(self):
        if not self.well_bottom <= self.screen_bottom < self.screen_top <= self.well_top:
            raise ValueError(f"Well {self.name}: the screen must lie within the well, with screen_top above bottom.")
        if self.diameter <= 0:
            raise ValueError(f"Well {self.name}: diameter must be positive.")


# Shapefile attribute names have at most 10 characters (e.g. "screen_bot").
_FIELDS = [f.name for f in fields(Well)]
_FULL_NAMES = {name[:10]: name for name in _FIELDS}


def array_radial(well: Well, n_wells: int, radius: float, start_azimuth: float = 0.0) -> list[Well]:
    """Copies of a well evenly spaced on a circle centred on the well location, numbered clockwise.

    Args:
        well: Well to copy; copies are named ``{well.name}_1``, ``{well.name}_2``, ...
        n_wells: Number of wells.
        radius: Radius of the circle [m].
        start_azimuth: Azimuth of the first well [°].

    Returns:
        List of wells.
    """
    if n_wells < 1 or radius <= 0:
        raise ValueError("n_wells must be at least 1 and radius must be positive.")
    wells = []
    for i in range(n_wells):
        azimuth = math.radians(start_azimuth + 360.0 * i / n_wells)
        x = well.x + radius * math.sin(azimuth)
        y = well.y + radius * math.cos(azimuth)
        wells.append(replace(well, name=f"{well.name}_{i + 1}", x=x, y=y))
    return wells


def array_linear(well: Well, n_wells: int, spacing: float, azimuth: float = 90.0) -> list[Well]:
    """Copies of a well evenly spaced along a straight line centred on the well location.

    Args:
        well: Well to copy; copies are named ``{well.name}_1``, ``{well.name}_2``, ...
        n_wells: Number of wells.
        spacing: Distance between neighbouring wells [m].
        azimuth: Direction of the line from the first to the last well [°]; 90 is west to east.

    Returns:
        List of wells.
    """
    if n_wells < 1 or spacing <= 0:
        raise ValueError("n_wells must be at least 1 and spacing must be positive.")
    direction = math.radians(azimuth)
    wells = []
    for i in range(n_wells):
        distance = (i - (n_wells - 1) / 2) * spacing
        x = well.x + distance * math.sin(direction)
        y = well.y + distance * math.cos(direction)
        wells.append(replace(well, name=f"{well.name}_{i + 1}", x=x, y=y))
    return wells


def read_wells(path: str | Path, well: Well | None = None) -> list[Well]:
    """Read wells from a point file (shapefile, GeoPackage, GeoJSON) or a CSV file.

    Each `Well` field is read from the attribute column of the same name and other columns are ignored;
    a CSV file also needs columns ``x`` and ``y``. Fields missing from the file are copied from ``well``; without a
    ``name`` column, wells are named ``{well.name}_1``, ``{well.name}_2``, ... Coordinates must be
    in metres (projected coordinate reference system); they are used as they are.

    Args:
        path: Path to the file.
        well: Well providing the fields missing from the file.

    Returns:
        List of wells, in file order.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    else:
        import geopandas

        table = geopandas.read_file(path)
        if table.crs is not None and table.crs.is_geographic:
            raise ValueError(f"{path.name}: coordinates are geographic; reproject them to a metric CRS.")
        if not (table.geom_type == "Point").all():
            raise ValueError(f"{path.name}: all geometries must be points.")
        table["x"], table["y"] = table.geometry.x, table.geometry.y
    table = table.rename(columns=lambda column: _FULL_NAMES.get(column.lower(), column.lower()))

    missing = [f for f in _FIELDS if f not in table.columns and f not in ("x", "y")]
    if missing and well is None:
        raise ValueError(f"{path.name}: no column for {', '.join(missing)}; give a well to copy them from.")
    for field in missing:
        table[field] = getattr(well, field)
    if "name" in missing:
        table["name"] = [f"{well.name}_{i + 1}" for i in range(len(table))]

    return [
        Well(name=str(row["name"]), **{f: float(row[f]) for f in _FIELDS if f != "name"}) for _, row in table.iterrows()
    ]


def write_wells(wells: list[Well], path: str | Path, crs: str | None = None) -> None:
    """Write wells with all their properties to a point file (shapefile, GeoPackage, GeoJSON) or a CSV file.

    Shapefile attribute names are truncated to 10 characters (e.g. ``screen_bot``); `read_wells` accepts them.

    Args:
        wells: List of wells.
        path: Path to the file; the format follows the extension.
        crs: Coordinate reference system of x and y, e.g. ``"EPSG:27572"``; not stored in CSV files.
    """
    path = Path(path)
    table = pd.DataFrame([asdict(w) for w in wells])
    if path.suffix.lower() == ".csv":
        table.to_csv(path, index=False)
    else:
        import geopandas

        points = geopandas.points_from_xy(table.pop("x"), table.pop("y"))
        if path.suffix.lower() == ".shp":
            table = table.rename(columns=lambda column: column[:10])
        geopandas.GeoDataFrame(table, geometry=points, crs=crs).to_file(path)
