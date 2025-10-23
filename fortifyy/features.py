
import numpy as np, pandas as pd
from typing import Union, Sequence
from .proprietary import compute_fri

ArrayLike = Union[float, int, Sequence[float], np.ndarray, pd.Series]


def _broadcast(values: ArrayLike, default: pd.Series) -> np.ndarray:
    length = len(default)
    default_arr = np.asarray(default, dtype=float)
    if values is None:
        return default_arr
    if isinstance(values, (int, float)):
        return np.full(length, float(values))
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return np.full(length, float(arr.item()))
    if arr.size != length:
        raise ValueError(f"Expected {length} values, received {arr.size}")
    return arr


def apply_scenario_adjustments(
    df: pd.DataFrame,
    wind_mph: ArrayLike = None,
    rain_in: ArrayLike = None,
) -> pd.DataFrame:
    d = df.copy()
    default_wind = d.get("historical_wind_mph", pd.Series([70.0] * len(d)))
    default_rain = d.get("historical_rain_in", pd.Series([10.0] * len(d)))
    wind_vals = _broadcast(wind_mph, default_wind)
    rain_vals = _broadcast(rain_in, default_rain)
    d["historical_wind_mph"] = wind_vals
    d["historical_rain_in"] = rain_vals
    d["wind_pressure_psf"] = 0.00256 * (d["historical_wind_mph"] ** 2)
    coeff_map = {"shingle": 0.90, "metal": 0.95, "tile": 0.85}
    d["runoff_coeff"] = (
        d.get("roof_material", pd.Series(["shingle"] * len(d)))
        .astype(str)
        .str.lower()
        .map(coeff_map)
        .fillna(0.90)
    )
    d["runoff_estimate_gal"] = (
        d["historical_rain_in"] * d["roof_area_sqft"] * 0.623 * d["runoff_coeff"]
    )
    return d


def enrich_proprietary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "wind_pressure_psf" not in d.columns or "runoff_estimate_gal" not in d.columns:
        d = apply_scenario_adjustments(d)
    d["fri"] = compute_fri(d)
    return d
