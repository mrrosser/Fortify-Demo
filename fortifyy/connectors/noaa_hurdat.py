
import os, math, requests, pandas as pd
from typing import Tuple, Optional, List

from fortifyy.firecrawl_client import fetch_lines_via_firecrawl, FirecrawlUnavailable

def default_hurdat_urls() -> List[str]:
    # Try env override, then a few likely candidates
    u = os.environ.get("HURDAT2_URL")
    if u:
        return [u]
    # Try recent years (current, last, previous) and generic file
    import datetime as dt
    y = dt.date.today().year
    years = [y, y-1, y-2, 2024, 2023]
    urls = []
    for yy in years:
        urls.append(f"https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-{yy}.txt")
    urls.append("https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024.txt")
    return urls

def fetch_hurdat2_text(urls: Optional[List[str]] = None, timeout: int = 30) -> str:
    urls = urls or default_hurdat_urls()
    last_err = None
    for u in urls:
        try:
            r = requests.get(u, timeout=timeout)
            if r.status_code == 200 and "AL011851" in r.text:
                return r.text
        except Exception as e:
            last_err = e
        try:
            lines = fetch_lines_via_firecrawl(u, timeout=timeout)
            text = "\n".join(lines)
            if "AL011851" in text:
                return text
        except FirecrawlUnavailable as fe:
            last_err = last_err or fe
    raise RuntimeError(f"Unable to fetch HURDAT2 from NHC. Last error: {last_err}")

def parse_hurdat2(text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    storms = []; rows = []; lines = text.splitlines(); i = 0
    while i < len(lines):
        header = lines[i].strip()
        if not header: i += 1; continue
        parts = [p.strip() for p in header.split(",")]
        if len(parts) >= 3 and parts[0][:2] in ("AL","CP","EP"):
            sid = parts[0]; name = parts[1].title(); n = int(parts[2])
            try:
                yy = int(sid[4:6]); year = 1900 + yy if yy > 50 else 2000 + yy
            except Exception:
                year = None
            storms.append({"storm_id": sid, "name": name, "year": year, "n_points": n})
            for j in range(1, n+1):
                if i+j >= len(lines): break
                pt = [p.strip() for p in lines[i+j].split(",")]
                if len(pt) >= 8:
                    date = pt[0]; time = pt[1]
                    record_id = pt[2]; status = pt[3]
                    lat = float(pt[4][:-1]) * (1 if pt[4][-1].upper()=="N" else -1)
                    lon = float(pt[5][:-1]) * (-1 if pt[5][-1].upper()=="W" else 1)
                    wind = int(pt[6]) if pt[6] != "" else None
                    pres = int(pt[7]) if pt[7] not in ("", "-999") else None
                    try: ts = pd.to_datetime(date + time, format="%Y%m%d%H%M")
                    except Exception: ts = pd.NaT
                    rows.append({"storm_id": sid, "datetime": ts, "record_id": record_id or "", "status": status or "", "lat": lat, "lon": lon, "wind_kt": wind, "pressure_mb": pres})
            i += n + 1
        else:
            i += 1
    return pd.DataFrame(storms), pd.DataFrame(rows).dropna(subset=["datetime"])

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def approx_max_wind_at_locations(points_df: pd.DataFrame, locs: pd.DataFrame,
                                 lat_col: str="lat", lon_col: str="lon",
                                 since: Optional[str]=None, until: Optional[str]=None,
                                 search_radius_km: float=200.0) -> pd.Series:
    df = points_df.copy()
    if since: df = df[df["datetime"] >= pd.to_datetime(since)]
    if until: df = df[df["datetime"] <= pd.to_datetime(until)]
    df = df.dropna(subset=["wind_kt"])
    if df.empty:
        return pd.Series([float("nan")]*len(locs), index=locs.index, name="hurdat_max_wind_mph")
    df["wind_mph"] = df["wind_kt"] * 1.15078

    vals = []
    for _, r in locs.iterrows():
        lat, lon = r[lat_col], r[lon_col]
        cand = df[(df["lat"].between(lat-3, lat+3)) & (df["lon"].between(lon-3, lon+3))]
        if cand.empty:
            vals.append(float("nan")); continue
        d = cand.apply(lambda x: haversine_km(lat, lon, x["lat"], x["lon"]), axis=1)
        mask = d <= search_radius_km
        if not mask.any():
            vals.append(float("nan")); continue
        atten = (1 - (d[mask] / search_radius_km)).clip(lower=0)
        max_wind = (cand.loc[mask, "wind_mph"] * atten).max()
        vals.append(float(max_wind))
    return pd.Series(vals, index=locs.index, name="hurdat_max_wind_mph")
