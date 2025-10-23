
from typing import Optional, Tuple
import requests, pandas as pd

from fortifyy.firecrawl_client import FirecrawlUnavailable, fetch_json_via_firecrawl
def query_fema_damage_assessments(feature_url: str, where: str = "1=1", bbox: Optional[Tuple[float,float,float,float]] = None, out_fields: str = "*", result_record_count: int = 2000) -> pd.DataFrame:
    url = feature_url.rstrip("/") + "/query"
    params = {"f":"json","where":where,"outFields":out_fields,"returnGeometry":"false","resultRecordCount":result_record_count}
    if bbox:
        minx, miny, maxx, maxy = bbox
        params.update({"geometry":f"{minx},{miny},{maxx},{maxy}","geometryType":"esriGeometryEnvelope","inSR":"4326","spatialRel":"esriSpatialRelIntersects"})
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
    except Exception as exc:
        full_url = requests.Request("GET", url, params=params).prepare().url
        try:
            js = fetch_json_via_firecrawl(full_url, timeout=30)
        except FirecrawlUnavailable as fe:
            raise RuntimeError(f"FEMA damage query failed for {full_url}: {exc}") from fe
    feats = js.get("features", [])
    rows = [f.get("attributes", {}) for f in feats]
    return pd.DataFrame(rows)
def summarize_zip_damage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    zip_col = None
    for cand in ["ZIP","zip","Zip","postalcode","PostalCode","POSTALCODE"]:
        if cand in df.columns: zip_col = cand; break
    if zip_col is None: return pd.DataFrame()
    d = df.copy(); d["zip"] = d[zip_col].astype(str).str.slice(0,5)
    cat = None
    for c in ["DamageCat","damagecat","damagelevel","Damage","Category","category"]:
        if c in d.columns: cat = c; break
    if cat is None:
        agg = d.groupby("zip").size().reset_index(name="fema_da_points")
    else:
        agg = d.groupby(["zip", d[cat].astype(str)]).size().unstack(fill_value=0)
        agg["fema_da_points"] = agg.sum(axis=1); agg = agg.reset_index()
    return agg
