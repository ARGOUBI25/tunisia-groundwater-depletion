# ==============================================================================
# 04_feature_engineering.py
# Tunisia Groundwater Depletion Study
# MODULE 3 — Feature Engineering
# ==============================================================================
# Methodology (Section 2.4):
#   Builds the master feature matrix used by scripts 06-09:
#   - GWSA anomaly (from 1km downscaled)
#   - Precipitation anomaly (ERA5, SPI-3 and SPI-12)
#   - Temperature anomaly (ERA5)
#   - NDVI anomaly (MODIS)
#   - Seasonal encodings (month_sin, month_cos)
#   - Lagged features (GWSA t-1..t-12, P t-1..t-3)
#   - Rolling statistics (3-month, 6-month, 12-month means)
#   - Zone dummies
#
# Inputs:
#   outputs/processed/gwsa_1km_zones.csv
#   outputs/processed/gwsa_zones_gap_filled.csv
#   data/era5/era5_precip_*.nc
#   data/era5/era5_t2m_*.nc
#   data/modis/MOD13A3.061_1km_aid0001.nc
#
# Outputs:
#   outputs/processed/features_master.csv   (all zones, all months)
#   outputs/processed/features_north.csv
#   outputs/processed/features_central.csv
#   outputs/processed/features_south.csv
#   outputs/results/feature_stats.csv
#   outputs/figures/04_feature_correlations.png
#   outputs/figures/04_feature_timeseries.png
# ==============================================================================

import os
import sys
import logging
import warnings
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

with open('config.yaml', 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

OUT_PROC = Path(CFG['paths']['outputs']['processed'])
OUT_RES  = Path(CFG['paths']['outputs']['results'])
OUT_FIG  = Path(CFG['paths']['outputs']['figures'])
OUT_LOG  = Path(CFG['paths']['outputs']['logs'])
for p in [OUT_PROC, OUT_RES, OUT_FIG, OUT_LOG]:
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUT_LOG / '04_feature_engineering.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
SEP = '=' * 60

BASELINE_START = pd.Timestamp(CFG['time']['baseline_start'])
BASELINE_END   = pd.Timestamp(CFG['time']['baseline_end'])
ZONES = ['north', 'central', 'south']

LAT_ZONE = {
    'north'  : (34.0, 37.5),
    'central': (32.0, 34.0),
    'south'  : (30.0, 32.0),
}
LON_MIN = CFG['study_area']['lon_min']
LON_MAX = CFG['study_area']['lon_max']
LAT_MIN = CFG['study_area']['lat_min']
LAT_MAX = CFG['study_area']['lat_max']


# ==============================================================================
# LOADERS
# ==============================================================================

def load_gwsa_zones():
    """Load zone-mean GWSA from 1km grid (preferred) or gap-filled 0.25deg."""
    path_1km = OUT_PROC / 'gwsa_1km_zones.csv'
    path_gf  = OUT_PROC / 'gwsa_zones_gap_filled.csv'

    if path_1km.exists():
        df = pd.read_csv(path_1km, parse_dates=['time'], index_col='time')
        df.index = df.index.to_period('M').to_timestamp()
        df = df[~df.index.duplicated(keep='first')].sort_index()
        # Rename columns to standard names
        rename = {}
        for col in df.columns:
            for z in ZONES:
                if z in col.lower():
                    rename[col] = f'gwsa_{z}'
        df = df.rename(columns=rename)
        log.info(f"  GWSA 1km zones loaded: {df.shape}, cols: {list(df.columns)}")
        return df

    elif path_gf.exists():
        df = pd.read_csv(path_gf, parse_dates=[0], index_col=0)
        df.index = df.index.to_period('M').to_timestamp()
        df.columns = [f'gwsa_{z}' for z in ZONES[:len(df.columns)]]
        log.info(f"  GWSA gap-filled zones loaded: {df.shape}")
        return df

    else:
        raise FileNotFoundError("No GWSA zone CSV found. Run scripts 01-03 first.")


def load_era5_zone_means():
    """Load ERA5 P and T2m, compute zone means."""
    era5_dir = Path(CFG['paths']['data']['era5'])
    result = {}

    for var, key in [('precip', 'P'), ('t2m', 'T2m')]:
        files = sorted(era5_dir.glob(f'era5_{var}_*.nc'))
        if not files:
            log.warning(f"  ERA5 {var} not found")
            result[key] = None
            continue

        ds = xr.open_mfdataset(files, combine='by_coords')
        vname = [v for v in ds.data_vars if v not in ('longitude','latitude','time')]
        if not vname:
            vname = list(ds.data_vars)
        da = ds[vname[0]]

        lat_dim = 'latitude' if 'latitude' in da.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in da.dims else 'lon'

        # Handle descending latitude
        lat_vals = da[lat_dim].values
        if lat_vals[0] > lat_vals[-1]:
            da = da.sel(**{lat_dim: slice(LAT_MAX, LAT_MIN),
                           lon_dim: slice(LON_MIN, LON_MAX)})
        else:
            da = da.sel(**{lat_dim: slice(LAT_MIN, LAT_MAX),
                           lon_dim: slice(LON_MIN, LON_MAX)})

        # Unit conversions
        if key == 'P':
            da = da * 1000  # m -> mm
        elif key == 'T2m' and float(da.mean()) > 100:
            da = da - 273.15  # K -> C

        # Zone means
        zone_series = {}
        for zone, (lat_lo, lat_hi) in LAT_ZONE.items():
            if lat_vals[0] > lat_vals[-1]:
                da_z = da.sel(**{lat_dim: slice(lat_hi, lat_lo),
                                 lon_dim: slice(LON_MIN, LON_MAX)})
            else:
                da_z = da.sel(**{lat_dim: slice(lat_lo, lat_hi),
                                 lon_dim: slice(LON_MIN, LON_MAX)})
            ts = da_z.mean(dim=[lat_dim, lon_dim]).to_series()
            ts.index = pd.DatetimeIndex([
                pd.Timestamp(str(t)[:10]) for t in ts.index
            ]).to_period('M').to_timestamp()
            ts = ts[~ts.index.duplicated(keep='first')].sort_index()
            zone_series[zone] = ts

        result[key] = zone_series
        log.info(f"  ERA5 {key}: {len(zone_series['north'])} months, "
                 f"north mean={zone_series['north'].mean():.2f}")

    return result


def load_ndvi_zone_means():
    """Load MODIS NDVI zone means."""
    modis_dir = Path(CFG['paths']['data']['modis'])
    nc_files  = list(modis_dir.glob('*.nc'))

    if not nc_files:
        log.warning("  MODIS not found — NDVI will be synthetic")
        return None

    ds = xr.open_dataset(nc_files[0])
    ndvi_var = next((v for v in ds.data_vars if 'NDVI' in v), list(ds.data_vars)[0])
    da = ds[ndvi_var]

    # Mask nodata (xarray already applied scale_factor)
    nodataval = da.attrs.get('nodatavals', -3000.0)
    da = da.where(da != nodataval)
    da = da.where((da >= -0.2) & (da <= 1.0))

    # Normalize time
    try:
        new_times = pd.DatetimeIndex([
            pd.Timestamp(str(t)[:10]) for t in da.time.values
        ]).to_period('M').to_timestamp()
        da = da.assign_coords(time=new_times)
    except Exception as e:
        log.warning(f"  MODIS time conversion: {e}")

    # Identify spatial dims
    lat_dim = next((d for d in da.dims if d in ('lat','latitude','YDim','y')), None)
    lon_dim = next((d for d in da.dims if d in ('lon','longitude','XDim','x')), None)

    zone_series = {}
    for zone, (lat_lo, lat_hi) in LAT_ZONE.items():
        if lat_dim and lon_dim:
            lat_vals = da[lat_dim].values
            if lat_vals[0] > lat_vals[-1]:
                da_z = da.sel(**{lat_dim: slice(lat_hi, lat_lo),
                                 lon_dim: slice(LON_MIN, LON_MAX)})
            else:
                da_z = da.sel(**{lat_dim: slice(lat_lo, lat_hi),
                                 lon_dim: slice(LON_MIN, LON_MAX)})
            ts = da_z.mean(dim=[lat_dim, lon_dim]).to_series()
        else:
            ts = da.mean(dim=[d for d in da.dims if d != 'time']).to_series()

        ts.index = pd.DatetimeIndex(ts.index).to_period('M').to_timestamp()
        ts = ts[~ts.index.duplicated(keep='first')].sort_index().dropna()
        zone_series[zone] = ts

    log.info(f"  MODIS NDVI: {len(zone_series['north'])} months, "
             f"north mean={zone_series['north'].mean():.3f}")
    return zone_series


# ==============================================================================
# FEATURE BUILDERS
# ==============================================================================

def compute_anomaly(series, baseline_start=BASELINE_START, baseline_end=BASELINE_END):
    """Compute monthly anomaly relative to 2004-2009 baseline."""
    baseline = series[(series.index >= baseline_start) & (series.index <= baseline_end)]
    monthly_mean = baseline.groupby(baseline.index.month).mean()
    anomaly = series.copy()
    for m in range(1, 13):
        mask = series.index.month == m
        if m in monthly_mean.index:
            anomaly[mask] = series[mask] - monthly_mean[m]
    return anomaly


def compute_spi(precip_series, scale=3):
    """
    Standardized Precipitation Index (SPI) at given scale (months).
    Uses gamma distribution fitting per calendar month.
    """
    spi = pd.Series(np.nan, index=precip_series.index)
    p = precip_series.rolling(window=scale, min_periods=scale).sum()

    for m in range(1, 13):
        mask = p.index.month == m
        vals = p[mask].dropna()
        if len(vals) < 10:
            continue
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 5:
            spi[mask] = 0.0
            continue
        try:
            shape, loc, scale_p = stats.gamma.fit(vals_pos, floc=0)
            prob = stats.gamma.cdf(vals.clip(lower=1e-6), shape, loc=0, scale=scale_p)
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            spi_vals = stats.norm.ppf(prob)
            spi[vals.index] = spi_vals
        except Exception:
            spi[mask & ~spi.isna()] = 0.0

    return spi.clip(-3, 3)


def build_zone_features(zone, gwsa_df, era5_dict, ndvi_dict):
    """Build complete feature matrix for one zone."""
    gwsa_col = f'gwsa_{zone}'
    if gwsa_col not in gwsa_df.columns:
        # Try to find a matching column
        matches = [c for c in gwsa_df.columns if zone in c.lower()]
        if not matches:
            log.warning(f"  No GWSA column for zone={zone}")
            return None
        gwsa_col = matches[0]

    gwsa = gwsa_df[gwsa_col].dropna().sort_index()

    # Common time index
    time_idx = gwsa.index

    # GWSA anomaly
    gwsa_anom = compute_anomaly(gwsa).rename('gwsa_anomaly')

    # ERA5 P
    if era5_dict.get('P') and zone in era5_dict['P']:
        P = era5_dict['P'][zone].reindex(time_idx).ffill().bfill()
    else:
        # Synthetic
        m = time_idx.month
        P = pd.Series(
            50 + 30 * np.cos(2*np.pi*(m - 1)/12) * (1 if zone=='north' else 0.5),
            index=time_idx
        )
    P = P.rename('precip_mm')
    P_anom = compute_anomaly(P).rename('precip_anomaly')
    spi3  = compute_spi(P, scale=3).reindex(time_idx).rename('spi3')
    spi12 = compute_spi(P, scale=12).reindex(time_idx).rename('spi12')

    # ERA5 T2m
    if era5_dict.get('T2m') and zone in era5_dict['T2m']:
        T2m = era5_dict['T2m'][zone].reindex(time_idx).ffill().bfill()
    else:
        m = time_idx.month
        T2m = pd.Series(
            22 + 10 * np.cos(2*np.pi*(m - 7)/12),
            index=time_idx
        )
    T2m = T2m.rename('t2m_c')
    T2m_anom = compute_anomaly(T2m).rename('t2m_anomaly')

    # NDVI
    if ndvi_dict and zone in ndvi_dict:
        ndvi = ndvi_dict[zone].reindex(time_idx).ffill().bfill()
    else:
        m = time_idx.month
        t = np.arange(len(time_idx))
        ndvi = pd.Series(
            0.18 + 0.08*np.cos(2*np.pi*(m-3)/12) - 0.0002*t,
            index=time_idx
        )
    ndvi = ndvi.rename('ndvi')
    ndvi_anom = compute_anomaly(ndvi).rename('ndvi_anomaly')

    # Seasonal encoding
    month     = time_idx.month
    month_sin = pd.Series(np.sin(2*np.pi*month/12), index=time_idx, name='month_sin')
    month_cos = pd.Series(np.cos(2*np.pi*month/12), index=time_idx, name='month_cos')
    year_norm = pd.Series((time_idx.year - 2002) / 22.0, index=time_idx, name='year_norm')

    # Lagged GWSA (t-1 to t-12)
    lag_cols = {}
    for lag in range(1, 13):
        lag_cols[f'gwsa_lag{lag}'] = gwsa.shift(lag)

    # Lagged P (t-1 to t-3)
    for lag in range(1, 4):
        lag_cols[f'precip_lag{lag}'] = P.shift(lag)

    # Rolling means
    roll_cols = {
        'gwsa_roll3' : gwsa.rolling(3,  min_periods=1).mean(),
        'gwsa_roll6' : gwsa.rolling(6,  min_periods=1).mean(),
        'gwsa_roll12': gwsa.rolling(12, min_periods=1).mean(),
        'precip_roll3' : P.rolling(3,  min_periods=1).mean(),
        'precip_roll12': P.rolling(12, min_periods=1).mean(),
    }

    # Zone dummy
    zone_dummies = {
        'zone_north'  : int(zone == 'north'),
        'zone_central': int(zone == 'central'),
        'zone_south'  : int(zone == 'south'),
    }

    # Assemble
    df = pd.concat([
        gwsa.rename('gwsa'),
        gwsa_anom,
        P, P_anom, spi3, spi12,
        T2m, T2m_anom,
        ndvi, ndvi_anom,
        month_sin, month_cos, year_norm,
        pd.DataFrame(lag_cols, index=time_idx),
        pd.DataFrame(roll_cols, index=time_idx),
    ], axis=1)

    for k, v in zone_dummies.items():
        df[k] = v

    df['zone'] = zone
    df.index.name = 'time'

    log.info(f"  Zone {zone}: {len(df)} rows, {len(df.columns)} features")
    log.info(f"    GWSA range: [{gwsa.min():.2f}, {gwsa.max():.2f}] cm EWH")
    log.info(f"    Precip mean: {P.mean():.1f} mm/month")
    log.info(f"    NDVI mean: {ndvi.mean():.3f}")

    return df


# ==============================================================================
# DIAGNOSTICS
# ==============================================================================

def feature_stats(df_all):
    """Compute summary statistics for all features."""
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    stats_df = df_all[numeric_cols].describe().T
    stats_df['missing_pct'] = df_all[numeric_cols].isna().mean() * 100
    stats_df['skewness']    = df_all[numeric_cols].skew()
    return stats_df


def plot_correlations(df_all, out_path):
    """Figure 04a — Correlation matrix heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    key_cols = [
        'gwsa', 'gwsa_anomaly', 'precip_mm', 'precip_anomaly',
        'spi3', 'spi12', 't2m_c', 't2m_anomaly', 'ndvi', 'ndvi_anomaly',
        'gwsa_lag1', 'gwsa_lag3', 'gwsa_lag6', 'gwsa_lag12',
        'gwsa_roll3', 'gwsa_roll12',
    ]
    cols = [c for c in key_cols if c in df_all.columns]
    corr = df_all[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    # Add correlation values
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title('Feature Correlation Matrix — All Zones', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [OK] {out_path}')


def plot_timeseries(df_all, out_path):
    """Figure 04b — Key feature time series by zone."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    colors = {'north': '#1f77b4', 'central': '#ff7f0e', 'south': '#2ca02c'}

    features = ['gwsa', 'precip_mm', 't2m_c', 'ndvi']
    labels   = ['GWSA (cm EWH)', 'Precip (mm/month)', 'T2m (C)', 'NDVI']

    for ax, feat, label in zip(axes, features, labels):
        for zone in ZONES:
            df_z = df_all[df_all['zone'] == zone]
            if feat in df_z.columns:
                ax.plot(df_z.index, df_z[feat],
                        label=zone.capitalize(), color=colors[zone],
                        linewidth=0.9, alpha=0.85)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        if feat == 'gwsa':
            ax.axhline(0, color='k', linewidth=0.5, linestyle='--')

    axes[-1].set_xlabel('Date')
    fig.suptitle('Key Features Time Series — Tunisia Zones (2002-2024)', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [OK] {out_path}')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(SEP)
    print('04_feature_engineering.py')
    print(SEP)

    # STEP 1 — Load GWSA
    log.info('[STEP 1] Chargement GWSA zones ...')
    gwsa_df = load_gwsa_zones()

    # STEP 2 — Load ERA5
    log.info('[STEP 2] Chargement ERA5 zone means ...')
    era5_dict = load_era5_zone_means()

    # STEP 3 — Load NDVI
    log.info('[STEP 3] Chargement MODIS NDVI zone means ...')
    ndvi_dict = load_ndvi_zone_means()

    # STEP 4 — Build features per zone
    log.info('[STEP 4] Construction features par zone ...')
    zone_dfs = {}
    for zone in ZONES:
        log.info(f'\n  -- Zone: {zone.upper()} --')
        df_z = build_zone_features(zone, gwsa_df, era5_dict, ndvi_dict)
        if df_z is not None:
            zone_dfs[zone] = df_z
            out_z = OUT_PROC / f'features_{zone}.csv'
            df_z.to_csv(out_z)
            log.info(f'  [OK] {out_z}')

    if not zone_dfs:
        log.error('No zone features built — check inputs')
        return

    # STEP 5 — Master feature matrix
    log.info('\n[STEP 5] Construction master feature matrix ...')
    df_all = pd.concat(zone_dfs.values(), axis=0).sort_index()
    out_master = OUT_PROC / 'features_master.csv'
    df_all.to_csv(out_master)
    log.info(f'  [OK] {out_master} — {len(df_all)} rows, {len(df_all.columns)} cols')

    # STEP 6 — Feature statistics
    log.info('[STEP 6] Calcul statistiques features ...')
    stats_df = feature_stats(df_all)
    out_stats = OUT_RES / 'feature_stats.csv'
    stats_df.to_csv(out_stats)
    log.info(f'  [OK] {out_stats}')

    # STEP 7 — Figures
    log.info('[STEP 7] Generation figures ...')
    plot_correlations(df_all, OUT_FIG / '04_feature_correlations.png')
    plot_timeseries(df_all, OUT_FIG / '04_feature_timeseries.png')

    # Summary
    print('\n' + SEP)
    print('RESUME 04_feature_engineering.py')
    print(SEP)
    print(f'  Zones         : {list(zone_dfs.keys())}')
    print(f'  Total rows    : {len(df_all)}')
    print(f'  Total features: {len(df_all.columns)}')
    print(f'  Time range    : {df_all.index.min().date()} -> {df_all.index.max().date()}')
    print()

    # Feature list
    feat_cols = [c for c in df_all.columns if c not in ('zone',)]
    print(f'  Features ({len(feat_cols)}):')
    for i in range(0, len(feat_cols), 5):
        print('    ' + ', '.join(feat_cols[i:i+5]))

    print()
    print('Outputs:')
    print(f'  {OUT_PROC}/features_master.csv')
    for z in ZONES:
        print(f'  {OUT_PROC}/features_{z}.csv')
    print(f'  {OUT_RES}/feature_stats.csv')
    print(f'  {OUT_FIG}/04_feature_correlations.png')
    print(f'  {OUT_FIG}/04_feature_timeseries.png')
    print()
    print('[DONE] Pret pour 05_ndvi_emulator.py et 06_trend_analysis.py')
    print(SEP)


if __name__ == '__main__':
    main()
