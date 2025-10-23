
import numpy as np, pandas as pd
from .proprietary import compute_fri
def apply_scenario_adjustments(df: pd.DataFrame, wind_mph: float, rain_in: float) -> pd.DataFrame:
    d = df.copy()
    d["historical_wind_mph"] = wind_mph
    d["historical_rain_in"] = rain_in
    d["wind_pressure_psf"] = 0.00256 * (d["historical_wind_mph"] ** 2)
    coeff_map = {"shingle":0.90,"metal":0.95,"tile":0.85}
    d["runoff_coeff"] = d.get("roof_material", pd.Series(["shingle"]*len(d))).astype(str).str.lower().map(coeff_map).fillna(0.90)
    d["runoff_estimate_gal"] = d["historical_rain_in"] * d["roof_area_sqft"] * 0.623 * d["runoff_coeff"]
    return d
def enrich_proprietary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "wind_pressure_psf" not in d.columns or "runoff_estimate_gal" not in d.columns:
        d = apply_scenario_adjustments(d, wind_mph=float(d.get("historical_wind_mph", pd.Series([70])).iloc[0]),
                                          rain_in=float(d.get("historical_rain_in", pd.Series([10])).iloc[0]))
    d["fri"] = compute_fri(d)
    return d
