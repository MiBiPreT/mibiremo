"""Time-dependent inputs [(t, value), ...] and their stress periods, shared by the field, column, and batch models."""

import numpy as np
import pandas as pd


def _stress_periods(inputs: dict, time_step: float | list, simulation_time: float) -> pd.DataFrame:
    """Stress periods starting at every change time of the time-dependent inputs (constants or [(t, value), ...]).

    Returns:
        DataFrame with start [s], duration [s], number of time steps (Δt ≤ time_step), and the value of each input.
    """
    time_step = _time_dependent(time_step)
    inputs = {name: _time_dependent(value) for name, value in inputs.items()}
    times = {t for values in [time_step, *inputs.values()] for t, _ in values}
    start = np.array(sorted(t for t in times if t < simulation_time))
    duration = np.diff([*start, simulation_time])
    n_time_steps = np.ceil(duration / _values_at(time_step, start) - 1e-9).astype(int)
    table = pd.DataFrame({"start": start, "duration": duration, "n_time_steps": n_time_steps})
    for name, values in inputs.items():
        table[name] = _values_at(values, start)
    return table


def _time_dependent(value):
    """Time-dependent input ``[(t, value), ...]`` from a time-dependent input or a constant."""
    if np.isscalar(value):
        return [(0.0, float(value))]
    times = [t for t, _ in value]
    if not times or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError(f"The times of a time-dependent input must start at 0 and increase: {times}.")
    return [(float(t), float(v)) for t, v in value]


def _values_at(time_dependent, times):
    """Values of a time-dependent input at the given times."""
    input_times, values = np.array(time_dependent).T
    return values[np.searchsorted(input_times, times, side="right") - 1]
