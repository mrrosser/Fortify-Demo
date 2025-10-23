
import pandas as pd, requests
from typing import Tuple

from fortifyy.firecrawl_client import (
    FirecrawlUnavailable,
    fetch_lines_via_firecrawl,
    fetch_json_via_firecrawl,
)

USGS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
PARAM_PRECIP_IN = "00045"

def find_precip_sites_in_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> pd.DataFrame:
    params = {"format":"rdb","bBox":f"{min_lon},{min_lat},{max_lon},{max_lat}","parameterCd":PARAM_PRECIP_IN,"hasDataTypeCd":"dv","siteStatus":"active"}
    prepared = requests.Request("GET", USGS_SITE_URL, params=params).prepare()
    full_url = prepared.url
    try:
        r = requests.get(prepared.url, timeout=30)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        text = r.text
    except Exception as exc:
        try:
            text = "\n".join(fetch_lines_via_firecrawl(full_url, timeout=30))
        except FirecrawlUnavailable as fe:
            raise RuntimeError(f"USGS site lookup failed for {full_url}: {exc}") from fe
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    hdr = [c.strip() for c in lines[0].split("\t")]
    data = [[c.strip() for c in ln.split("\t")] for ln in lines[2:] if ln.strip()]
    df = pd.DataFrame(data, columns=hdr)
    keep = ["agency_cd","site_no","station_nm","site_tp_cd","dec_lat_va","dec_long_va"]
    df = df[[c for c in keep if c in df.columns]]
    df["dec_lat_va"] = pd.to_numeric(df["dec_lat_va"], errors="coerce")
    df["dec_long_va"] = pd.to_numeric(df["dec_long_va"], errors="coerce")
    return df

def fetch_daily_precip(site_no: str, start_date: str, end_date: str) -> pd.DataFrame:
    params = {"format":"json","sites":site_no,"parameterCd":PARAM_PRECIP_IN,"startDT":start_date,"endDT":end_date}
    prepared = requests.Request("GET", USGS_DV_URL, params=params).prepare()
    full_url = prepared.url
    try:
        r = requests.get(full_url, timeout=30)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        js = r.json()
    except Exception as exc:
        try:
            js = fetch_json_via_firecrawl(full_url, timeout=30)
        except FirecrawlUnavailable as fe:
            raise RuntimeError(f"USGS precipitation lookup failed for {full_url}: {exc}") from fe
    rows = []
    for ts in js.get("value",{}).get("timeSeries",[]):
        for v in ts.get("values",[]):
            for val in v.get("value",[]):
                rows.append({"dateTime": val.get("dateTime"), "value_in": val.get("value")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["dateTime"]).dt.date
        df["value_in"] = pd.to_numeric(df["value_in"], errors="coerce")
    return df

def nearest_precip_total(lat: float, lon: float, start_date: str, end_date: str, bbox: Tuple[float,float,float,float]) -> float:
    sites = find_precip_sites_in_bbox(*bbox)
    if sites.empty:
        return float("nan")
    sites["dist"] = ((sites["dec_lat_va"] - lat)**2 + (sites["dec_long_va"] - lon)**2)**0.5
    site = sites.sort_values("dist").iloc[0]
    df = fetch_daily_precip(site["site_no"], start_date, end_date)
    return float(df["value_in"].sum()) if not df.empty else float("nan")
