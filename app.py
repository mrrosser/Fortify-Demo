
import io, json, os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import datetime as dt
import altair as alt

from fortifyy.config import DISPLAY_NAME, CLASS_TARGET, REG_TARGET, NUM_FEATURES, CAT_FEATURES, MAP_DEFAULTS
from fortifyy.data import load_sample_data, validate_and_prepare
from fortifyy.model import (
    build_classification_pipeline,
    build_calibrated_classifier, build_regression_pipeline,
    evaluate_classifier, evaluate_regressor, top_permutation_importances
)
from fortifyy.features import apply_scenario_adjustments, enrich_proprietary
from fortifyy.viz import make_risk_map


def render_metric_cards(metric_dict, layout):
    cols = st.columns(len(layout))
    for col, (label, key, fmt) in zip(cols, layout):
        if not metric_dict or key not in metric_dict or pd.isna(metric_dict[key]):
            col.metric(label, "--")
        else:
            col.metric(label, fmt.format(metric_dict[key]))


st.set_page_config(page_title=f"{DISPLAY_NAME} - Local Demo (LIVE)", layout="wide")
st.title(f"{DISPLAY_NAME} - Local Demo (LIVE)")
st.caption("Roof damage probability and runoff with Flood/Wind/Hail inputs (NOAA/USGS/FEMA).")
firecrawl_key = os.environ.get("FIRECRAWL_API_KEY")
if not firecrawl_key:
    try:
        firecrawl_key = st.secrets.get("FIRECRAWL_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        firecrawl_key = None
    if firecrawl_key:
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_key

with st.sidebar:
    pw_env = os.environ.get('FORTIFYY_DEMO_PASSWORD')
    if pw_env:
        pw_in = st.text_input('Access password', type='password')
        if pw_in != pw_env:
            st.error('Enter valid password to continue.')
            st.stop()
    if not firecrawl_key:
        st.caption("Tip: set FIRECRAWL_API_KEY to enable Firecrawl fallback for NOAA/USGS/FEMA.")

    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    use_sample = st.checkbox("Use bundled sample dataset", value=True if not uploaded else False)

    st.header("Public Data Enrichment (LIVE)")
    enrich_toggle = st.checkbox("Enrich dataset with NOAA/USGS/FEMA", value=True)
    ed1 = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=365*10))
    ed2 = st.date_input("End date", value=dt.date.today())
    fema_url = st.text_input("FEMA Damage Assessments FeatureServer URL (optional)", value=os.environ.get("FEMA_DA_FEATURE_URL", ""))

    st.header("Scenario")
    scenario_wind = st.slider("Storm wind speed (mph)", 30, 170, 95, 5)
    scenario_rain = st.slider("Storm rainfall (inches)", 0, 30, 12, 1)
    counterfactual_toggle = st.selectbox("Counterfactual fortified status", ["none", "force_fortified", "force_unfortified"], index=0)

    st.header("Training")
    train_button = st.button("Train / Re-train Models")
    export_models = st.checkbox("Enable model artifact download", value=True)

    st.header("Map settings")
    map_radius = st.slider("Point radius (meters)", 10, 200, MAP_DEFAULTS["radius_meters"], 5)
    map_threshold = st.slider("High-risk threshold (%)", 0, 100, 40, 1)

# Load data
if uploaded and not use_sample:
    raw = pd.read_csv(uploaded)
    st.success(f"Loaded uploaded CSV with shape {raw.shape}")
else:
    raw = load_sample_data()
    st.info(f"Loaded bundled sample dataset with shape {raw.shape}")

df, issues = validate_and_prepare(raw)
df = enrich_proprietary(df)

# Optional enrichment
if enrich_toggle:
    from fortifyy.connectors.enrich import enrich_with_public_data
    bbox = (-90.5, 29.6, -89.6, 30.2)  # Greater New Orleans
    fema_feature_url = fema_url.strip() or None
    with st.spinner('Fetching NOAA/USGS/FEMA features...'):
        df = enrich_with_public_data(df, start_date=str(ed1), end_date=str(ed2), bbox=bbox, fema_feature_url=fema_feature_url)
    st.success('Enriched with public data.')
    if any(c for c in df.columns if c.startswith("_enrich_warn_")):
        with st.expander("Enrichment warnings", expanded=False):
            warns = [f"{c}: {df[c].iloc[0]}" for c in df.columns if c.startswith("_enrich_warn_")]
            for wmsg in warns: st.warning(wmsg)

if issues:
    with st.expander("Data validation notes", expanded=False):
        for msg in issues:
            st.warning(msg)

# Train or reuse
if "clf" not in st.session_state or "reg" not in st.session_state or train_button:
    clf = build_calibrated_classifier(NUM_FEATURES, CAT_FEATURES)
    reg = build_regression_pipeline(NUM_FEATURES, CAT_FEATURES)
    clf_for_importance = build_classification_pipeline(NUM_FEATURES, CAT_FEATURES)

    X = df[NUM_FEATURES + CAT_FEATURES]
    y_clf = df[CLASS_TARGET]
    y_reg = df[REG_TARGET]

    from sklearn.model_selection import train_test_split
    Xc_train, Xc_val, yc_train, yc_val = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    clf.fit(Xc_train, yc_train)

    Xr_train, Xr_cal, yr_train, yr_cal = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg.fit(Xr_train, yr_train)
    cal_pred = reg.predict(Xr_cal)
    cal_resid = (np.abs(yr_cal - cal_pred))
    q90 = float(np.quantile(cal_resid, 0.90))

    clf_for_importance.fit(X, y_clf)

    clf_metrics = evaluate_classifier(clf, X, y_clf)
    reg_metrics = evaluate_regressor(reg, X, y_reg)

    importance_df = None
    importance_error = None
    try:
        importance_df = top_permutation_importances(
            clf_for_importance, X, y_clf, NUM_FEATURES + CAT_FEATURES, n_top=12
        )
        if not importance_df.empty:
            total = importance_df["importance"].abs().sum()
            if total > 0:
                importance_df["importance_pct"] = (importance_df["importance"].abs() / total) * 100.0
    except Exception as e:
        importance_error = str(e)

    st.session_state['conformal_q90'] = q90
    st.session_state["clf"] = clf
    st.session_state["reg"] = reg
    st.session_state["clf_metrics"] = clf_metrics
    st.session_state["reg_metrics"] = reg_metrics
    st.session_state["clf_importances"] = importance_df
    st.session_state["clf_importances_error"] = importance_error

    st.success("Models (re)trained.")

clf = st.session_state.get("clf", None)
reg = st.session_state.get("reg", None)
if clf is None or reg is None:
    st.stop()

clf_metrics = st.session_state.get("clf_metrics", {})
reg_metrics = st.session_state.get("reg_metrics", {})
importance_df = st.session_state.get("clf_importances")
importance_error = st.session_state.get("clf_importances_error")

with st.expander("Model metrics & explainability", expanded=True):
    st.subheader("Damage classifier")
    render_metric_cards(
        clf_metrics,
        [
            ("ROC AUC", "roc_auc", "{:.3f}"),
            ("Avg precision", "avg_precision", "{:.3f}"),
            ("Accuracy", "accuracy", "{:.3f}"),
            ("Brier score", "brier", "{:.3f}"),
            ("ECE (10 bins)", "ece_10", "{:.3f}"),
        ],
    )
    if importance_df is not None and not importance_df.empty and importance_df["importance"].abs().sum() > 0:
        display_df = importance_df.copy()
        if "importance_pct" not in display_df.columns:
            total = display_df["importance"].abs().sum()
            display_df["importance_pct"] = (display_df["importance"].abs() / total * 100.0) if total > 0 else 0.0
        display_df = display_df.fillna(0)
        display_df["importance"] = display_df["importance"].round(4)
        display_df["importance_pct"] = display_df["importance_pct"].round(1)
        chart_data = display_df.sort_values("importance", ascending=True)
        chart = (
            alt.Chart(chart_data)
            .mark_bar(color="#4e79a7")
            .encode(
                x=alt.X("importance:Q", title="Permutation importance"),
                y=alt.Y("feature:N", sort=alt.Sort(field="importance", order="ascending")),
                tooltip=[alt.Tooltip("feature:N", title="Feature"),
                         alt.Tooltip("importance:Q", title="Importance", format=".4f"),
                         alt.Tooltip("importance_pct:Q", title="Share (%)", format=".1f")]
            )
        )
        st.altair_chart(chart, use_container_width=True)
        pretty_df = display_df.rename(columns={"feature": "Feature", "importance": "Importance", "importance_pct": "Share (%)"})
        st.dataframe(pretty_df)
    elif importance_error:
        st.caption(f"Importance calc skipped: {importance_error}")
    else:
        st.caption("Feature importance was flat (all zeros). Try retraining with enrichment enabled or a larger dataset.")

    st.subheader("Runoff regressor")
    render_metric_cards(
        reg_metrics,
        [
            ("MAE (gal)", "mae", "{:.0f}"),
            ("RMSE (gal)", "rmse", "{:.0f}"),
        ],
    )
    st.caption("Runoff interval uses split-conformal (90%) and appears in the predictions table.")

# Scenario inference
scenario_df = apply_scenario_adjustments(df.copy(), wind_mph=scenario_wind, rain_in=scenario_rain)
if counterfactual_toggle == "force_fortified":
    scenario_df["fortified"] = 1
elif counterfactual_toggle == "force_unfortified":
    scenario_df["fortified"] = 0
scenario_df = enrich_proprietary(scenario_df)

X_s = scenario_df[NUM_FEATURES + CAT_FEATURES]
probs = clf.predict_proba(X_s)[:, 1]
runoff = reg.predict(X_s)
q90 = st.session_state.get('conformal_q90', 0.0)
runoff_lo = runoff - q90
runoff_hi = runoff + q90

scenario_df["pred_damage_prob"] = probs
scenario_df["pred_runoff_gal"] = runoff
scenario_df["pred_runoff_lo"] = runoff_lo
scenario_df["pred_runoff_hi"] = runoff_hi

# KPIs
high_risk = (scenario_df["pred_damage_prob"] >= (map_threshold/100)).sum()
avg_prob = float(np.mean(scenario_df["pred_damage_prob"]) * 100)
total_runoff_kgal = float(np.sum(scenario_df["pred_runoff_gal"]) / 1000.0)

k1, k2, k3 = st.columns(3)
k1.metric("Avg damage risk", f"{avg_prob:.1f}%")
k2.metric(f"High-risk parcels (>={map_threshold}%)", f"{high_risk}")
k3.metric("Total runoff (kgal)", f"{total_runoff_kgal:.1f}")

# Map
st.subheader("Risk Map")
layer = make_risk_map(scenario_df, radius_meters=map_radius)
view_state = pdk.ViewState(latitude=29.9511, longitude=-90.0715, zoom=10, pitch=0)
r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{tooltip}"})
st.pydeck_chart(r, use_container_width=True)
with st.container():
    st.markdown(
        f"""
        <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background-color:rgb(255,64,64);"></span>
                <span style="font-size:0.9rem;">Higher risk (>= {map_threshold}%)</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background-color:rgb(255,200,80);"></span>
                <span style="font-size:0.9rem;">Medium risk (~20-{map_threshold}% range)</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background-color:rgb(64,200,64);"></span>
                <span style="font-size:0.9rem;">Lower risk (&lt;~20%)</span>
            </div>
            <span style="font-size:0.85rem;color:#6c757d;">Point size follows the radius slider; hover for parcel-level details.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Table & downloads
with st.expander("Predictions table", expanded=False):
    cols = ["parcel_id", "lat", "lon", "zip", "fortified",
            "pred_damage_prob", "pred_runoff_gal", "pred_runoff_lo", "pred_runoff_hi"]
    st.dataframe(scenario_df[cols].sort_values("pred_damage_prob", ascending=False))

pred_csv = scenario_df.copy()
pred_csv["pred_damage_prob_pct"] = pred_csv["pred_damage_prob"] * 100
csv_bytes = pred_csv.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", data=csv_bytes, file_name="fortifyy_predictions.csv", mime="text/csv")

if export_models:
    clf_bytes = io.BytesIO()
    reg_bytes = io.BytesIO()
    joblib.dump(st.session_state["clf"], clf_bytes)
    joblib.dump(st.session_state["reg"], reg_bytes)
    clf_bytes.seek(0); reg_bytes.seek(0)
    st.download_button("Download damage model (joblib)", data=clf_bytes, file_name="damage_model.joblib")
    st.download_button("Download runoff model (joblib)", data=reg_bytes, file_name="runoff_model.joblib")

st.caption("Demo only. Replace with real data and calibrations for production.")
