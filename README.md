
# Fortifyy Local Demo (LIVE) — Flood/Wind/Hail inputs → Roof Risk

This is a local, VS Code–friendly Streamlit app with:
- Proprietary **FORTIFYY_RISK_INDEX (FRI)**
- Calibrated classifier for damage probability; conformal intervals for runoff
- **Live connectors**: NOAA HURDAT2 (wind tracks), USGS NWIS (precip), optional FEMA ArcGIS (damage pts)

A Streamlit *User Guide* page (available under the **☰ menu → User Guide**) summarizes the workflow for non-technical reviewers.

## Quickstart (macOS/Linux)
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
# optional gate
export FORTIFYY_DEMO_PASSWORD="demo"
# optional FEMA FeatureServer layer 0 (Historical Damage Assessments)
# export FEMA_DA_FEATURE_URL="https://.../FeatureServer/0"
streamlit run app.py
```

## Quickstart (Windows PowerShell)
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
$env:FORTIFYY_DEMO_PASSWORD="demo"
# $env:FEMA_DA_FEATURE_URL="https://.../FeatureServer/0"
streamlit run app.py
```

If NOAA HURDAT2 URL changes, set (optional):
```bash
export HURDAT2_URL="https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025.txt"
```

### Firecrawl-assisted data fetching
For more reliable demos (and to avoid public API hiccups), you can route all enrichment calls through Firecrawl:

```powershell
# Windows PowerShell
$env:FIRECRAWL_API_KEY="fc-...your-key..."
```
```bash
# macOS/Linux
export FIRECRAWL_API_KEY="fc-...your-key..."
```

Optional overrides:
- `FIRECRAWL_API_URL` to point at a self-hosted Firecrawl instance.
- `HURDAT2_URL`, `FEMA_DA_FEATURE_URL`, or a narrowed NWIS bounding box if you have curated mirrors.

When the key is set, the app transparently falls back to Firecrawl if direct NOAA/USGS/FEMA requests fail, keeping the map populated for client demos.
If both live and Firecrawl calls fail (e.g., the NOAA text file is temporarily unavailable), the app drops back to the bundled `sample_data/hurdat2_sample.txt` so the demo still renders winds without breaking.
The USGS connector behaves the same way, inserting the cached `sample_data/usgs_precip_sample.json` total if no active gauges respond.

Open http://localhost:8501 in your browser.
