import streamlit as st

st.set_page_config(page_title="Fortifyy Demo – User Guide", layout="wide")
st.title("Fortifyy Demo – User Guide")
st.caption("Quick reference for clients and teammates exploring the local demo.")

st.markdown(
    """
## 1. Getting Started
- **Access** – Enter the demo password (default `demo`) if prompted.
- **Data selection** – Upload your CSV or keep the bundled sample dataset checked to explore synthetic Greater New Orleans roofs.
- **Training** – Click **Train / Re-train Models** after changing data or enrichment settings to refresh predictions and metrics.
- **Scenario sliders** – Adjust storm wind/rainfall and fortified status to see how risk responds in real time.

## 2. Understanding the Dashboard
- **Enrichment status badges** summarize which public feeds (NOAA, USGS, FEMA) are live, cached, or unavailable.
- **Gauge cards** show average damage probability, the share of parcels above the high-risk threshold, and total runoff volume.
- **Map legend** explains the color gradient (red = higher risk, green = lower) and honors the radius slider.
- **Metrics & Explainability** lists classifier/regression performance plus permutation importances for top features.
- **Predictions table** lets you sort parcels and download CSVs of the full scenario output.

## 3. Data Sources & Fallbacks
- **NOAA HURDAT2** – Live API → Firecrawl fallback → bundled sample (`sample_data/hurdat2_sample.txt`) so winds display even during outages.
- **USGS precipitation** – Live NWIS gauges; if none respond or no daily values exist, a cached rainfall total (`sample_data/usgs_precip_sample.json`) is inserted and flagged.
- **FEMA damage assessments** – Optional FeatureServer URL; when provided, totals by ZIP are merged in.

Set these environment variables (or Streamlit secrets) to customize data access:
```
FIRECRAWL_API_KEY        # enables Firecrawl fallback
HURDAT2_URL              # override to a curated hurricane archive
FEMA_DA_FEATURE_URL      # default FeatureServer endpoint
FORTIFYY_DEMO_PASSWORD   # protects the app UI
```

Need more detail? The project README in the repository covers installation commands and advanced configuration tips.
"""
)
