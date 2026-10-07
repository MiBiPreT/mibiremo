"""Plotting functions"""

import math
import matplotlib.pyplot as plt
import numpy as np
from flopy.plot import PlotCrossSection
from flopy.plot import PlotMapView


def map_view(model, values, layer=0, label=None, ax=None, **kwargs):
    """Plots a map of `FieldModel`, wells, and a selected value field.

    Args:
        model: `FieldModel` after `build` or `run`.
        values: Array (layer, row, column), e.g. `model.concentration(time)`, `model.head(time)`, or a zone.
        layer: Layer, 0 at the top.
        label: Label of the colour bar.
        ax: matplotlib axes; default: a new figure.
        **kwargs: Arguments of flopy `plot_array`, e.g. `cmap`, `vmin`, `vmax`.

    Returns:
        matplotlib axes.
    """
    ax = ax or plt.subplots()[1]
    mesh = PlotMapView(modelgrid=model.grid, ax=ax, layer=layer).plot_array(np.asarray(values, dtype=float), **kwargs)
    plt.colorbar(mesh, ax=ax, label=label)
    wells(ax, model.wells, model.flow_rates, ms=7)
    ax.set(xlabel="x [m]", ylabel="y [m]")
    ax.ticklabel_format(useOffset=False, style="plain")
    return ax


def cross_section(model, values, line, label=None, ax=None, **kwargs):
    """Cross section of `FieldModel` along a line.

    Args:
        model: `FieldModel` after `build` or `run`.
        values: Array (layer, row, column), e.g. `model.concentration(time)`.
        line: End points (x0, y0) and (x1, y1) of the line [m].
        label: Label of the colour bar.
        ax: matplotlib axes; default: a new figure.
        **kwargs: Arguments of flopy `plot_array`, e.g. `cmap`, `vmin`, `vmax`.

    Returns:
        matplotlib axes.
    """
    ax = ax or plt.subplots()[1]
    section = PlotCrossSection(modelgrid=model.grid, ax=ax, line={"line": line}, geographic_coords=True)
    mesh = section.plot_array(np.asarray(values, dtype=float), **kwargs)
    plt.colorbar(mesh, ax=ax, label=label)
    (x0, y0), (x1, y1) = line
    length = math.hypot(x1 - x0, y1 - y0)
    for w in model.wells:  # wells within half a fine cell of the line
        along = ((w.x - x0) * (x1 - x0) + (w.y - y0) * (y1 - y0)) / length
        across = abs((w.x - x0) * (y1 - y0) - (w.y - y0) * (x1 - x0)) / length
        if across <= model.grid_spacing_at_wells / 2 and 0 <= along <= length:
            position = w.x if section.direction == "x" else w.y
            ax.plot([position, position], [w.screen_bottom, w.screen_top], color="k", lw=2)
            ax.annotate(w.name, (position, w.screen_top), textcoords="offset points", xytext=(3, -10), fontsize=8)
    ax.set(xlabel=f"{section.direction} [m]", ylabel="elevation [m]")
    ax.ticklabel_format(useOffset=False, style="plain")
    return ax


def wells(ax, wells, flow_rates=None, labels=True, **kwargs):
    """Map view of wells.

    Args:
        ax: matplotlib axes.
        wells: Wells.
        flow_rates: Flow rate Q [m³ s⁻¹] by well name, constant or schedule; a well that ever injects is an injection
            well.
        labels: If True, write the well names.
        **kwargs: Arguments of matplotlib `plot`, e.g. `ms`.
    """
    flow_rates = flow_rates or {}
    for w in wells:
        q = flow_rates.get(w.name, 0.0)
        q = [value for _, value in q] if isinstance(q, list) else [q]
        marker, color = ("^", "tab:blue") if max(q) > 0 else ("o", "tab:red") if min(q) < 0 else ("s", "0.6")
        style = {"marker": marker, "color": color, "mec": "k", "ms": 9}
        ax.plot(w.x, w.y, ls="", **(style | kwargs))
        if labels:
            ax.annotate(w.name, (w.x, w.y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_aspect("equal")


def flow_arrow(ax, azimuth):
    """Plots the direction of groundwater flow.

    Args:
        ax: matplotlib axes.
        azimuth: Direction of the groundwater flow, clockwise from north [°].
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    length = 0.15 * min(x1 - x0, y1 - y0)
    start = (x0 + 0.1 * (x1 - x0), y0 + 0.1 * (y1 - y0))
    end = (start[0] + length * math.sin(math.radians(azimuth)), start[1] + length * math.cos(math.radians(azimuth)))
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "0.3", "lw": 1.5})
    ax.annotate("groundwater flow", start, textcoords="offset points", xytext=(0, -12), fontsize=8, color="0.3")
