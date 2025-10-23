
from datetime import date, timedelta
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
    try:
        txt = fetch_hurdat2_text(); _, pts = parse_hurdat2(txt)
        d["hurdat_max_wind_mph"] = approx_max_wind_at_locations(pts, d, "lat", "lon", since=start_date, until=end_date)
    except Exception as e:
        d["hurdat_max_wind_mph"] = float("nan"); d["_enrich_warn_hurdat2"] = str(e)
    # USGS precip (centroid site)
    try:
        total = nearest_precip_total(d["lat"].mean(), d["lon"].mean(), start_date, end_date, bbox=bbox)
        d["usgs_rain_total_in"] = total
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
