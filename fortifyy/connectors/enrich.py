
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from .noaa_hurdat import fetch_hurdat2_text, parse_hurdat2, approx_max_wind_at_locations
from .usgs_precip import nearest_precip_total
from .fema_da import query_fema_damage_assessments, summarize_zip_damage

def enrich_with_public_data(df: pd.DataFrame, start_date: str = None, end_date: str = None, bbox = (-90.5, 29.6, -89.6, 30.2), fema_feature_url: str = None) -> pd.DataFrame:
    d = df.copy()
    if start_date is None or end_date is None:
        end = date.today(); start = end - timedelta(days=365*10)
        start_date, end_date = start.isoformat(), end.isoformat()
    # HURDAT2
    hurdat_warn = None
    try:
        txt = fetch_hurdat2_text()
    except Exception as e:
        hurdat_warn = f"Live HURDAT2 fetch failed ({e})"
        sample_path = Path(__file__).resolve().parents[1] / "sample_data" / "hurdat2_sample.txt"
        if sample_path.exists():
            txt = sample_path.read_text()
            if hurdat_warn:
                hurdat_warn += "; using bundled sample"
        else:
            txt = None
    if txt:
        try:
            _, pts = parse_hurdat2(txt)
            d["hurdat_max_wind_mph"] = approx_max_wind_at_locations(pts, d, "lat", "lon", since=start_date, until=end_date)
        except Exception as e:  # noqa: BLE001
            d["hurdat_max_wind_mph"] = float("nan")
            hurdat_warn = hurdat_warn or ""
            hurdat_warn = (hurdat_warn + f"; parsing error {e}").strip("; ")
    else:
        d["hurdat_max_wind_mph"] = float("nan")
        if not hurdat_warn:
            hurdat_warn = "HURDAT2 unavailable and no local sample found"
    if hurdat_warn:
        d["_enrich_warn_hurdat2"] = hurdat_warn
    # USGS precip (centroid site)
    try:
        total, precip_note = nearest_precip_total(d["lat"].mean(), d["lon"].mean(), start_date, end_date, bbox=bbox)
        d["usgs_rain_total_in"] = total
        if precip_note:
            d["_enrich_warn_usgs"] = precip_note
    except Exception as e:
        d["usgs_rain_total_in"] = float("nan"); d["_enrich_warn_usgs"] = str(e)
    # FEMA DA (optional)
    if fema_feature_url:
        try:
            da = query_fema_damage_assessments(fema_feature_url, where="1=1", bbox=bbox)
            if not da.empty and "zip" in d.columns:
                d["zip"] = d["zip"].astype(str).str.slice(0,5)
                agg = summarize_zip_damage(da)
                if not agg.empty: d = d.merge(agg, on="zip", how="left")
        except Exception as e:
            d["_enrich_warn_fema"] = str(e)
    return d
