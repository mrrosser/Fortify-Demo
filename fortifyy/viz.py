
import numpy as np, pandas as pd, pydeck as pdk
def make_risk_map(df: pd.DataFrame, radius_meters: int = 40) -> pdk.Layer:
    probs = df["pred_damage_prob"].clip(0,1).values
    colors = np.zeros((len(probs), 3), dtype=int)
    colors[:,0] = (probs * 255).astype(int)          # R
    colors[:,1] = ((1 - probs) * 255).astype(int)    # G
    colors[:,2] = 50
    data = pd.DataFrame({
        "lat": df["lat"].values,
        "lon": df["lon"].values,
        "color": [ [int(r),int(g),int(b)] for r,g,b in colors ],
        "radius": radius_meters,
        "tooltip": df.apply(lambda r: f"Parcel: {r['parcel_id']}\\nRisk: {r['pred_damage_prob']*100:.1f}%\\nRunoff: {r['pred_runoff_gal']:.0f} gal (±{(r['pred_runoff_hi']-r['pred_runoff_gal']):.0f})", axis=1)
    })
    return pdk.Layer("ScatterplotLayer", data=data, get_position='[lon, lat]', get_color='color', get_radius='radius', pickable=True, opacity=0.5)
