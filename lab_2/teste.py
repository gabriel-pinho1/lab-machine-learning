DATE = "date"
TARGET = "chlorophyll_a_mg_m3"
for p in range (10,15):
    for q in range(8,15):
        Lag_spec = {
        TARGET: list(range(1, p)),
        "sst_c": list(range(0, p)),
        "par_umol_m2_s": list(range(0, q)),
        "nitrate_umol_l": list(range(0, p)),
        "wind_speed_m_s": list(range(0, q)),
        "upwelling_index": list(range(0, p)),
        "mixed_layer_depth_m": list(range(0, q)),
        "salinity_psu": list(range(0, q)),
        "current_speed_m_s": list(range(0, q)),
        "river_discharge_index": list(range(0, q)),
        "cloud_fraction": list(range(0, q)),
        "surface_pressure_hpa": list(range(0, q)),
        "turbidity_ntu": list(range(0, q)),
        }