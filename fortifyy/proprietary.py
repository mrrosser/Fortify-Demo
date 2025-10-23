
import numpy as np, pandas as pd
def _uplift_coeff(roof_pitch_deg, roof_material):
    m = roof_material.astype(str).str.lower().fillna("shingle")
    base = 1.0 + (roof_pitch_deg.clip(0, 60) / 60.0)
    mat_adj = np.where(m.eq("metal"), 1.1, np.where(m.eq("tile"), 0.9, 1.0))
    return base * mat_adj
def _runoff_load_coeff(rain_in, roof_area_sqft, material):
    coeff = material.astype(str).str.lower().map({"metal":0.95,"tile":0.85,"shingle":0.90}).fillna(0.90)
    gallons = rain_in.clip(0) * roof_area_sqft.clip(100, None) * 0.623 * coeff
    return gallons / 10000.0
def _tree_fall_risk(tree_density, building_height_ft):
    return (tree_density.clip(0,1) * (building_height_ft.clip(5,80) / 30.0))
def _flood_vuln(flood_zone, elevation_ft):
    zone_weight = flood_zone.astype(str).map({"AE":1.0,"A":0.7,"X":0.2}).fillna(0.5)
    elev_term = (10.0 - elevation_ft.clip(-5, 30)) / 10.0
    return (zone_weight * elev_term).clip(0, 2.0)
def _exposure_proxy(historical_wind_mph, lat, lon, zip_code):
    wind_term = (historical_wind_mph.clip(20, 180) - 20.0) / 100.0
    jitter = ((lat - lat.mean()).abs() + (lon - lon.mean()).abs()) * 2.0
    return (wind_term + jitter).clip(0, 2.0)
def _aging_penalty(roof_age_yrs): return (roof_age_yrs.clip(0, 60) / 30.0).clip(0, 2.0)
def _fortification_benefit(fortified): return np.where(fortified.astype(int) == 1, 0.6, 0.0)
def compute_fri(df: pd.DataFrame) -> pd.Series:
    uplift = _uplift_coeff(df["roof_pitch_deg"], df["roof_material"])
    runoff = _runoff_load_coeff(df["historical_rain_in"], df["roof_area_sqft"], df["roof_material"])
    tree_risk = _tree_fall_risk(df["tree_density"], df["building_height_ft"])
    flood = _flood_vuln(df["flood_zone"], df["elevation_ft"])
    exposure = _exposure_proxy(df["historical_wind_mph"], df["lat"], df["lon"], df["zip"])
    aging = _aging_penalty(df["roof_age_yrs"])
    fort_benefit = _fortification_benefit(df["fortified"])
    w = {"uplift":0.28,"runoff":0.18,"tree":0.14,"flood":0.14,"exposure":0.16,"aging":0.12,"fort":1.00,"bias":0.35}
    raw = (w["bias"] + w["uplift"]*uplift + w["runoff"]*runoff + w["tree"]*tree_risk + w["flood"]*flood + w["exposure"]*exposure + w["aging"]*aging - w["fort"]*fort_benefit)
    return raw.clip(0, 4.0)
