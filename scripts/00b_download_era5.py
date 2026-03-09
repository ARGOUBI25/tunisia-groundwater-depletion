"""
00b_download_era5.py
Téléchargement automatique ERA5 via API CDS
Tunisia Groundwater Study
"""

import cdsapi
import os
import yaml

# Config
with open("../config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

OUT_DIR = cfg["paths"]["data"]["era5"]
os.makedirs(OUT_DIR, exist_ok=True)

# Client CDS
# Prerequis : fichier C:\Users\RTX\.cdsapirc contenant :
#   url: https://cds.climate.copernicus.eu/api
#   key: TON-API-KEY

c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="bd52f35f-8c44-44c3-94c4-e26f87d2ab2f"
)

VARIABLES = {
    "total_precipitation": "precip",
    "2m_temperature":      "t2m"
}

YEARS  = [str(y) for y in range(2002, 2025)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

AREA = [
    cfg["study_area"]["lat_max"],  # North
    cfg["study_area"]["lon_min"],  # West
    cfg["study_area"]["lat_min"],  # South
    cfg["study_area"]["lon_max"],  # East
]

# Download
for var_long, var_short in VARIABLES.items():
    out_file = os.path.join(OUT_DIR, f"era5_{var_short}_monthly_2002_2024.nc")

    if os.path.exists(out_file):
        print(f"[SKIP] {out_file} already exists.")
        continue

    print(f"[DOWNLOAD] {var_long} ...")
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable":     [var_long],
            "year":         YEARS,
            "month":        MONTHS,
            "time":         "00:00",
            "area":         AREA,
            "format":       "netcdf",
            "grid":         [0.25, 0.25],
        },
        out_file
    )
    print(f"[OK] Saved to {out_file}")

print("\n[DONE] ERA5 download complete.")
print(f"Files saved in: {OUT_DIR}")