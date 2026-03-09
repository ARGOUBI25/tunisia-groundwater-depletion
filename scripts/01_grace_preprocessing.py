"""
01_grace_preprocessing.py
═══════════════════════════════════════════════════════════════════════════════
Module 1 — GRACE/GRACE-FO Preprocessing
- Chargement des 3 mascons (CSR, JPL, GFZ)
- Calcul moyenne d'ensemble TWSA + σ_TWSA
- Application masque terre (>50% land fraction → 148 pixels)
- Isolation GWSA = TWSA - SM - SW - SWE (GLDAS NOAH + VIC)
- Propagation incertitude σ_GWSA²
- Outputs : gwsa_ensemble.nc, gwsa_uncertainty.nc
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import yaml
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Fix working directory (fonctionne peu importe comment PyCharm lance) ──────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)

# ── 0. Config ─────────────────────────────────────────────────────────────────
with open(os.path.join(BASE, "config.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

GRACE_DIR    = cfg["paths"]["data"]["grace"]
NOAH_DIR     = cfg["paths"]["data"]["gldas_noah"]
VIC_DIR      = cfg["paths"]["data"]["gldas_vic"]
OUT_DIR      = cfg["paths"]["outputs"]["processed"]
LOG_DIR      = cfg["paths"]["outputs"]["logs"]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BASELINE_START = cfg["time"]["baseline_start"]
BASELINE_END   = cfg["time"]["baseline_end"]
LAND_THRESHOLD = cfg["grace"]["land_mask_threshold"]

# Bounding box Tunisie
LAT_MIN = cfg["study_area"]["lat_min"]
LAT_MAX = cfg["study_area"]["lat_max"]
LON_MIN = cfg["study_area"]["lon_min"]
LON_MAX = cfg["study_area"]["lon_max"]

print("=" * 60)
print("01_grace_preprocessing.py")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Chargement des 3 mascons GRACE
# ══════════════════════════════════════════════════════════════════════════════

def load_mascon(filepath, var_name, label):
    """
    Charge un fichier mascon NetCDF, extrait la variable TWSA,
    sélectionne la zone Tunisie, retourne un DataArray mensuel.
    """
    print(f"  [LOAD] {label} : {filepath}")
    ds = xr.open_dataset(filepath)

    # Afficher les variables disponibles pour debug
    print(f"         Variables: {list(ds.data_vars)}")
    print(f"         Coords   : {list(ds.coords)}")

    # Extraire la variable TWSA (nom peut varier selon mascon)
    if var_name not in ds:
        # Essayer des noms alternatifs communs
        candidates = ["lwe_thickness", "twsa", "TWSA", "lwe_thickness_csr",
                      "lwe_thickness_jpl", "lwe_thickness_gfz", "lwe"]
        for c in candidates:
            if c in ds:
                var_name = c
                break
        else:
            raise ValueError(f"Variable TWSA non trouvée dans {filepath}. "
                             f"Variables disponibles: {list(ds.data_vars)}")

    da = ds[var_name]

    # Normaliser les noms de coordonnées
    rename_map = {}
    for coord in da.coords:
        if coord.lower() in ["latitude", "lat"]:
            rename_map[coord] = "lat"
        if coord.lower() in ["longitude", "lon"]:
            rename_map[coord] = "lon"
        if coord.lower() in ["time", "date"]:
            rename_map[coord] = "time"
    if rename_map:
        da = da.rename(rename_map)

    # Clip sur la Tunisie
    da = da.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    # Convertir en cm EWH si nécessaire (souvent en m)
    if da.attrs.get("units", "") in ["m", "meters"]:
        da = da * 100.0
        da.attrs["units"] = "cm EWH"

    da.name = f"twsa_{label.lower()}"
    print(f"         Shape: {da.shape}, Period: {da.time.values[0]} → {da.time.values[-1]}")
    return da


print("\n[STEP 1] Chargement des mascons GRACE ...")

# Adapter les noms de fichiers à ce que tu as téléchargé
mascon_files = {
    "JPL": os.path.join(GRACE_DIR, "GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc"),
}

twsa_dict = {}
for label, fpath in mascon_files.items():
    if not os.path.exists(fpath):
        print(f"  [WARN] Fichier non trouvé: {fpath}")
        print(f"         Vérifie le nom exact du fichier dans data/grace/")
        continue
    twsa_dict[label] = load_mascon(fpath, "lwe_thickness", label)

if len(twsa_dict) == 0:
    raise FileNotFoundError("Aucun fichier mascon trouvé dans data/grace/. "
                            "Télécharge les données GRACE d'abord.")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Alignement temporel et calcul moyenne d'ensemble
# ══════════════════════════════════════════════════════════════════════════════

print("\n[STEP 2] Calcul de la moyenne d'ensemble TWSA ...")

# Aligner sur la même grille temporelle commune
time_arrays = [da.time.values for da in twsa_dict.values()]
common_times = time_arrays[0]
for t in time_arrays[1:]:
    common_times = np.intersect1d(common_times, t)

print(f"  Période commune: {common_times[0]} → {common_times[-1]}")
print(f"  Nombre de pas de temps: {len(common_times)}")

# Sélectionner la période commune
twsa_aligned = {}
for label, da in twsa_dict.items():
    twsa_aligned[label] = da.sel(time=common_times)

# Empiler les mascons en un seul Dataset
twsa_stack = xr.concat(
    [twsa_aligned[k] for k in twsa_aligned],
    dim=pd.Index(list(twsa_aligned.keys()), name="mascon")
)

# Moyenne d'ensemble et écart-type inter-mascon
twsa_mean = twsa_stack.mean(dim="mascon")
twsa_std  = twsa_stack.std(dim="mascon")

twsa_mean.name = "twsa_mean"
twsa_std.name  = "twsa_sigma"

twsa_mean.attrs["units"]       = "cm EWH"
twsa_mean.attrs["long_name"]   = "TWSA ensemble mean (CSR+JPL+GFZ)"
twsa_std.attrs["units"]        = "cm EWH"
twsa_std.attrs["long_name"]    = "TWSA inter-mascon std (sigma_TWSA)"

print(f"  TWSA mean shape: {twsa_mean.shape}")
print(f"  sigma_TWSA mean: {float(twsa_std.mean()):.3f} cm EWH")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Masque terre (>50% land fraction → 148 pixels)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[STEP 3] Application du masque terre ...")

# Calcul du masque à partir de la variance temporelle
# Les pixels ocean ont TWSA ~0 et très faible variance
twsa_var = twsa_mean.var(dim="time")
land_mask = twsa_var > twsa_var.quantile(0.10)  # exclure pixels quasi-constants

n_pixels_total = int(land_mask.size)
n_pixels_land  = int(land_mask.sum())

print(f"  Pixels dans bounding box : {n_pixels_total}")
print(f"  Pixels terre (masque)    : {n_pixels_land}")
print(f"  Attendu ~148 pixels")

# Appliquer le masque
twsa_mean_masked = twsa_mean.where(land_mask)
twsa_std_masked  = twsa_std.where(land_mask)

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Calcul des anomalies par rapport à la baseline 2004-2009
# ══════════════════════════════════════════════════════════════════════════════

print("\n[STEP 4] Calcul des anomalies TWSA (baseline 2004-2009) ...")

baseline = twsa_mean_masked.sel(
    time=slice(BASELINE_START, BASELINE_END)
).mean(dim="time")

twsa_anomaly = twsa_mean_masked - baseline
twsa_anomaly.name = "twsa_anomaly"
twsa_anomaly.attrs["units"]     = "cm EWH"
twsa_anomaly.attrs["long_name"] = "TWSA anomaly relative to 2004-2009 baseline"
twsa_anomaly.attrs["baseline"]  = f"{BASELINE_START} to {BASELINE_END}"

print(f"  TWSA anomaly range: {float(twsa_anomaly.min()):.2f} to "
      f"{float(twsa_anomaly.max()):.2f} cm EWH")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Chargement GLDAS NOAH (SM, SW, SWE)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[STEP 5] Chargement GLDAS NOAH (SM, SW, SWE) ...")

def load_gldas(directory, model_name):
    """
    Charge tous les fichiers GLDAS NetCDF d'un dossier,
    calcule SM total, SW, SWE, retourne un Dataset.
    """
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".nc") or f.endswith(".nc4")
    ])

    if len(files) == 0:
        print(f"  [WARN] Aucun fichier GLDAS trouvé dans {directory}")
        return None

    print(f"  {model_name}: {len(files)} fichiers trouvés")
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False, engine="netcdf4")
    print(f"  Coords disponibles: {list(ds.coords)}")
    print(f"  Lat range: {float(ds.lat.min())} à {float(ds.lat.max())}")
    print(f"  Lon range: {float(ds.lon.min())} à {float(ds.lon.max())}")
    # Clip sur la Tunisie
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"

    ds = ds.where(
        (ds[lat_name] >= LAT_MIN) & (ds[lat_name] <= LAT_MAX) &
        (ds[lon_name] >= LON_MIN) & (ds[lon_name] <= LON_MAX),
        drop=True
    )

    # Calculer SM total (somme des 4 couches de sol en kg/m² → cm EWH)
    # 1 kg/m² = 0.1 cm EWH
    sm_vars = [v for v in ds.data_vars if "SoilMoi" in v or "soil_moist" in v.lower()]
    if sm_vars:
        sm_total = sum(ds[v] for v in sm_vars) * 0.1  # kg/m² → cm EWH
        sm_total.name = f"sm_{model_name.lower()}"
        sm_total.attrs["units"] = "cm EWH"
    else:
        print(f"  [WARN] Variables SM non trouvées. Vars: {list(ds.data_vars)[:10]}")
        sm_total = None

    # SWE (Snow Water Equivalent)
    swe_vars = [v for v in ds.data_vars if "SWE" in v or "swe" in v.lower()]
    swe = ds[swe_vars[0]] * 0.1 if swe_vars else xr.zeros_like(sm_total)
    swe.name = "swe"
    swe.attrs["units"] = "cm EWH"

    # SW (Surface Water — approximé par surface runoff accumulé)
    sw_vars = [v for v in ds.data_vars if "Qs_acc" in v or "Qs" in v]
    sw = ds[sw_vars[0]] * 0.1 if sw_vars else xr.zeros_like(sm_total)
    sw.name = "sw"
    sw.attrs["units"] = "cm EWH"

    return xr.Dataset({"sm": sm_total, "swe": swe, "sw": sw})


noah_ds = load_gldas(NOAH_DIR, "NOAH")
vic_ds = None  # VIC à télécharger plus tard

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Isolation GWSA et propagation d'incertitude
# ══════════════════════════════════════════════════════════════════════════════

print("\n[STEP 6] Isolation GWSA = TWSA - SM - SW - SWE ...")

if noah_ds is not None:
    # Aligner temporellement GLDAS sur TWSA
    noah_interp = noah_ds.interp(
        time=twsa_anomaly.time,
        lat=twsa_anomaly.lat,
        lon=twsa_anomaly.lon,
        method="linear"
    )
    print(f"  DEBUG twsa_anomaly time[0]: {twsa_anomaly.time.values[0]}")
    print(f"  DEBUG noah_interp time[0]: {noah_interp.time.values[0]}")
    print(f"  DEBUG twsa shape: {twsa_anomaly.shape}")
    print(f"  DEBUG noah sm shape: {noah_interp['sm'].shape}")
    print(f"  DEBUG sm values sample: {noah_interp['sm'].isel(time=0).values}")
    # Anomalies GLDAS par rapport à la même baseline
    def compute_anomaly(da, baseline_start, baseline_end):
        baseline_mean = da.sel(
            time=slice(baseline_start, baseline_end)
        ).mean(dim="time")
        return da - baseline_mean

    sm_anom  = compute_anomaly(noah_interp["sm"],  BASELINE_START, BASELINE_END)
    sw_anom  = compute_anomaly(noah_interp["sw"],  BASELINE_START, BASELINE_END)
    swe_anom = compute_anomaly(noah_interp["swe"], BASELINE_START, BASELINE_END)

    # GWSA = TWSA - SM - SW - SWE (Eq. 1 du papier)
    gwsa = twsa_anomaly - sm_anom - sw_anom - swe_anom
    gwsa.name = "gwsa"
    gwsa = gwsa.compute()
    gwsa.attrs["units"]     = "cm EWH"
    gwsa.attrs["long_name"] = "Groundwater Storage Anomaly"
    gwsa.attrs["equation"]  = "GWSA = TWSA - SM - SW - SWE"

    gwsa_vals = gwsa.values
    import numpy as np

    print(f"  GWSA range: {np.nanmin(gwsa_vals):.2f} to {np.nanmax(gwsa_vals):.2f} cm EWH")
    print(f"  GWSA non-NaN values: {np.sum(~np.isnan(gwsa_vals))}")

    # ── Propagation d'incertitude (Eq. 2 du papier) ──────────────────────────
    print("\n[STEP 6b] Propagation d'incertitude σ_GWSA ...")

    sigma_twsa = twsa_std_masked

    # σ_SM : différence NOAH vs VIC (incertitude structurelle du modèle)
    if vic_ds is not None:
        vic_interp = vic_ds.interp(time=twsa_anomaly.time, method="linear")
        vic_sm_anom = compute_anomaly(vic_interp["sm"], BASELINE_START, BASELINE_END)
        sigma_sm = abs(sm_anom - vic_sm_anom)
    else:
        sigma_sm = sm_anom * 0.15  # 15% par défaut si VIC non disponible
        print("  [WARN] VIC non disponible, σ_SM estimé à 15% de SM")

    # σ_SW et σ_SWE : incertitude fixe (SWE négligeable en Tunisie)
    sigma_sw  = sw_anom  * 0.20   # 20% pour surface water
    sigma_swe = swe_anom * 0.10   # SWE negligible en Tunisie (<0.1 cm EWH)

    # σ²_GWSA = σ²_TWSA + σ²_SM + σ²_SW + σ²_SWE
    sigma_gwsa_sq = (sigma_twsa**2 + sigma_sm**2 +
                     sigma_sw**2  + sigma_swe**2)
    sigma_gwsa = np.sqrt(sigma_gwsa_sq)
    sigma_gwsa.name = "sigma_gwsa"
    sigma_gwsa.attrs["units"]     = "cm EWH"
    sigma_gwsa.attrs["long_name"] = "GWSA uncertainty (1-sigma)"

    print(f"  σ_GWSA mean: {float(sigma_gwsa.mean()):.3f} cm EWH")
    print(f"  σ_GWSA max : {float(sigma_gwsa.max()):.3f} cm EWH")

else:
    print("  [WARN] GLDAS NOAH non disponible.")
    print("         GWSA = TWSA (sans soustraction SM/SW/SWE)")
    gwsa = twsa_anomaly.copy()
    gwsa.name = "gwsa"
    sigma_gwsa = twsa_std_masked.copy()
    sigma_gwsa.name = "sigma_gwsa"

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Agrégation par zone hydroclimatique
# ══════════════════════════════════════════════════════════════════════════════

print("\n[STEP 7] Agrégation par zone hydroclimatique ...")

zones = {
    "north":   {"lat": (34.0, 37.5), "lon": (7.5, 11.6)},
    "central": {"lat": (32.0, 34.0), "lon": (7.5, 11.6)},
    "south":   {"lat": (30.0, 32.0), "lon": (7.5, 11.6)},
}

zone_gwsa = {}
for zone_name, bounds in zones.items():
    zone_da = gwsa.sel(
        lat=slice(bounds["lat"][0], bounds["lat"][1]),
        lon=slice(bounds["lon"][0], bounds["lon"][1])
    ).mean(dim=["lat", "lon"])
    zone_da.name = f"gwsa_{zone_name}"
    zone_gwsa[zone_name] = zone_da
    print(f"  Zone {zone_name:8s}: mean GWSA = {float(zone_da.mean()):.3f} cm EWH")

# DataFrame mensuel par zone
df_zones = pd.DataFrame(
    {zone: zone_gwsa[zone].values for zone in zone_gwsa},
    index=pd.DatetimeIndex(gwsa.time.values)
)
df_zones.index.name = "date"
df_zones.columns = ["gwsa_north_cm", "gwsa_central_cm", "gwsa_south_cm"]

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 8 — Sauvegarde des outputs
# ══════════════════════════════════════════════════════════════════════════════

print("\n[STEP 8] Sauvegarde des outputs ...")

# Dataset principal
ds_out = xr.Dataset({
    "twsa_mean":    twsa_mean_masked,
    "twsa_sigma":   twsa_std_masked,
    "twsa_anomaly": twsa_anomaly,
    "gwsa":         gwsa,
    "sigma_gwsa":   sigma_gwsa,
})

ds_out.attrs["title"]       = "GRACE/GRACE-FO GWSA — Tunisia 2002-2024"
ds_out.attrs["institution"] = "Tunisia Groundwater Study"
ds_out.attrs["created"]     = datetime.now().strftime("%Y-%m-%d %H:%M")
ds_out.attrs["mascons"]     = "CSR RL06M + JPL RL06M + GFZ RL06"
ds_out.attrs["baseline"]    = "2004-2009"

out_nc = os.path.join(OUT_DIR, "gwsa_ensemble.nc")
ds_out.to_netcdf(out_nc)
print(f"  [OK] {out_nc}")

# GWSA par zone en CSV
out_csv = os.path.join(OUT_DIR, "gwsa_zones_monthly.csv")
df_zones.to_csv(out_csv)
print(f"  [OK] {out_csv}")

# Statistiques de base
stats = df_zones.describe()
out_stats = os.path.join(OUT_DIR, "gwsa_statistics.csv")
stats.to_csv(out_stats)
print(f"  [OK] {out_stats}")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 9 — Figure de contrôle
# ══════════════════════════════════════════════════════════════════════════════

print("\n[STEP 9] Génération figure de contrôle ...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

colors = {"north": "#2196F3", "central": "#FF9800", "south": "#F44336"}
labels = {"north": "Northern Zone", "central": "Central Zone", "south": "Southern Zone"}

for ax, (zone, color) in zip(axes, colors.items()):
    col = f"gwsa_{zone}_cm"
    ax.plot(df_zones.index, df_zones[col],
            color=color, linewidth=1.2, label=labels[zone])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.fill_between(df_zones.index, df_zones[col], 0,
                    where=df_zones[col] < 0,
                    alpha=0.3, color=color, label="Deficit")
    ax.set_ylabel("GWSA (cm EWH)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvspan(
        pd.Timestamp("2017-07-01"),
        pd.Timestamp("2018-05-01"),
        alpha=0.15, color="gray", label="Gap GRACE"
    )

axes[0].set_title("GRACE/GRACE-FO GWSA by Hydroclimatic Zone — Tunisia 2002–2024",
                  fontsize=12, fontweight="bold")
axes[-1].set_xlabel("Date", fontsize=10)

plt.tight_layout()
fig_path = os.path.join(cfg["paths"]["outputs"]["figures"], "01_gwsa_timeseries.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK] {fig_path}")

# ── Résumé final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RÉSUMÉ 01_grace_preprocessing.py")
print("=" * 60)
print(f"  Mascons chargés   : {list(twsa_dict.keys())}")
print(f"  Période           : {common_times[0]} → {common_times[-1]}")
print(f"  Pixels terre      : {n_pixels_land} / {n_pixels_total}")
print(f"  GWSA zones        : North / Central / South")
print(f"\nOutputs:")
print(f"  {out_nc}")
print(f"  {out_csv}")
print(f"  {out_stats}")
print(f"  {fig_path}")
print("\n[DONE] Prêt pour 02_gap_filling.py")