"""Field installation: MODFLOW 6 groundwater flow and solute transport.

- `structured_grid`: structured (DIS) grid centred on the wells
"""

import numpy as np
from flopy.discretization import StructuredGrid

MAX_GROWTH_FACTOR = 1.5  # largest ratio between the widths of neighbouring cells outside the fine zone


def structured_grid(
    wells, domain_size, top, bottom, n_layers, grid_spacing, grid_spacing_at_wells, refinement_margin=5.0
):
    """Structured grid centred on the wells and refined around them.

    Args:
        wells: Wells used to refine the grid (e.g., pumped wells).
        domain_size: Size of the domain along x and y [m].
        top: Elevation of the top of the aquifer [m].
        bottom: Elevation of the bottom of the aquifer [m].
        n_layers: Number of layers.
        grid_spacing: Largest cell width Δx [m].
        grid_spacing_at_wells: Cell width Δx around the wells [m].
        refinement_margin: Distance from the outermost wells to the edge of the fine cells [m].

    Returns:
        flopy StructuredGrid in the coordinates of the wells; row 1 is the northernmost.
    """
    if not wells:
        raise ValueError("At least one well is needed.")
    if grid_spacing_at_wells > grid_spacing:
        raise ValueError("grid_spacing_at_wells must not exceed grid_spacing.")
    x = [w.x for w in wells]
    y = [w.y for w in wells]
    xorigin = (min(x) + max(x) - domain_size[0]) / 2
    yorigin = (min(y) + max(y) - domain_size[1]) / 2
    spacing = {"coarse": grid_spacing, "fine": grid_spacing_at_wells, "margin": refinement_margin}
    delr = _cell_widths(x, xorigin, xorigin + domain_size[0], **spacing)
    delc = _cell_widths(y, yorigin, yorigin + domain_size[1], **spacing)[::-1]  # MODFLOW rows run north to south
    n_rows, n_columns = len(delc), len(delr)
    botm = np.linspace(top, bottom, n_layers + 1)[1:]
    return StructuredGrid(
        delc=delc,
        delr=delr,
        top=np.full((n_rows, n_columns), float(top)),
        botm=np.repeat(botm, n_rows * n_columns).reshape(n_layers, n_rows, n_columns),
        xoff=xorigin,
        yoff=yorigin,
    )


def _cell_widths(coordinates, start, end, coarse, fine, margin):
    """Cell widths from start to end along one axis: fine cells centred on the wells, growing to coarse ones."""
    coordinates = np.sort(coordinates)
    # Wells closer than `fine` share a block of fine cells; a single well has one cell centred on it.
    clusters = np.split(coordinates, np.flatnonzero(np.diff(coordinates) >= fine) + 1)
    widths = []
    for previous, cluster in zip([None, *clusters], clusters):
        block = _equal_widths(cluster[-1] - cluster[0] + fine, fine)
        if previous is not None:
            gap = cluster[0] - previous[-1] - fine
            filler = _equal_widths(gap, fine)
            if not filler:  # a gap narrower than fine / 2 is shared by the cells on either side
                widths[-1] += gap / 2
                block[0] += gap / 2
            widths += filler
        widths += block
    n_margin = round(margin / fine)
    widths = [fine] * n_margin + widths + [fine] * n_margin

    # Outside the fine zone, the widths grow towards the domain boundaries.
    left = coordinates[0] - fine / 2 - n_margin * fine - start
    right = end - coordinates[-1] - fine / 2 - n_margin * fine
    if min(left, right) < 0:
        raise ValueError("The domain is too small for the wells and the refinement margin.")
    if left < fine / 2:  # too short for a cell of its own
        widths[0] += left
        left = 0.0
    if right < fine / 2:
        widths[-1] += right
        right = 0.0
    return np.array(_growing_widths(left, fine, coarse)[::-1] + widths + _growing_widths(right, fine, coarse))


def _equal_widths(length, spacing):
    """Equal widths, as close as possible to spacing, that fill length; none if length < spacing / 2."""
    n = round(length / spacing)
    return [length / n] * n if n else []


def _growing_widths(length, fine, coarse):
    """Widths growing from fine by MAX_GROWTH_FACTOR up to coarse, scaled down to fill length exactly."""
    widths = []
    while sum(widths) < length:
        widths.append(min(fine * MAX_GROWTH_FACTOR ** (len(widths) + 1), coarse))
    return [w * length / sum(widths) for w in widths]
