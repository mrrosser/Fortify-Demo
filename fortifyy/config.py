
DISPLAY_NAME = "Fortifyy Predict"
CLASS_TARGET = "damage"
REG_TARGET = "runoff_volume_gal"
REQUIRED_COLUMNS = ["parcel_id","lat","lon","zip","fortified","roof_age_yrs","roof_material","elevation_ft","flood_zone","tree_density","building_height_ft","roof_pitch_deg","historical_wind_mph","historical_rain_in","roof_area_sqft","damage","runoff_volume_gal"]
BASE_NUM_FEATURES = ["roof_age_yrs","elevation_ft","tree_density","building_height_ft","roof_pitch_deg","historical_wind_mph","historical_rain_in","roof_area_sqft","fortified"]
CAT_FEATURES = ["roof_material","flood_zone","zip"]
PROPRIETARY_FEATURES = ["fri","wind_pressure_psf","runoff_estimate_gal"]
PUBLIC_NUM_FEATURES = ["hurdat_max_wind_mph","usgs_rain_total_in"]
NUM_FEATURES = BASE_NUM_FEATURES + PROPRIETARY_FEATURES + PUBLIC_NUM_FEATURES
MAP_DEFAULTS = {"radius_meters": 40}
