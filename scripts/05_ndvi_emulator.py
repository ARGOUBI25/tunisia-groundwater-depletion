# ==============================================================================
# 05_ndvi_emulator.py
# Tunisia Groundwater Depletion Study
# MODULE 4 — NDVI Emulator (Random Forest)
# ==============================================================================
# Methodology (Section 2.5):
#   Train RF to predict NDVI from ERA5 P + T2m (+ LAI if available)
#   so that future NDVI can be estimated from CMIP6 P+T2m projections
#   under SSP2-4.5 and SSP5-8.5.
#
#   Train: 2002-2022 | Val: 2023-2024
#   Features: P_t, P_t-1, P_t-2, T2m_t, T2m_t-1, month_sin, month_cos
#   Target: NDVI (zone mean from MODIS)
#
# Inputs:
#   outputs/processed/features_master.csv
#   data/cmip6/  (optional — if available)
#
# Outputs:
#   outputs/models/ndvi_emulator_[zone].pkl
#   outputs/processed/ndvi_emulated_historical.csv
#   outputs/processed/ndvi_emulated_ssp245.csv   (if CMIP6 available)
#   outputs/processed/ndvi_emulated_ssp585.csv
#   outputs/results/ndvi_emulator_metrics.csv
#   outputs/figures/05_ndvi_emulator_[zone].png
# ==============================================================================

import os
import sys
import logging
import warnings
import yaml
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

with open('config.yaml', 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

OUT_PROC = Path(CFG['paths']['outputs']['processed'])
OUT_MOD  = Path(CFG['paths']['outputs']['models'])
OUT_RES  = Path(CFG['paths']['outputs']['results'])
OUT_FIG  = Path(CFG['paths']['outputs']['figures'])
OUT_LOG  = Path(CFG['paths']['outputs']['logs'])
for p in [OUT_PROC, OUT_MOD, OUT_RES, OUT_FIG, OUT_LOG]:
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUT_LOG / '05_ndvi_emulator.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
SEP = '=' * 60

TRAIN_END = pd.Timestamp(CFG['ndvi_emulator']['train_end'])
VAL_START = pd.Timestamp(CFG['ndvi_emulator']['val_start'])
N_EST     = CFG['ndvi_emulator']['n_estimators']
MAX_D     = CFG['ndvi_emulator']['max_depth']
ZONES     = ['north', 'central', 'south']
RANDOM_STATE = CFG['random_state']


# ==============================================================================
# FEATURE BUILDER FOR NDVI EMULATOR
# ==============================================================================

def build_ndvi_features(df_zone):
    """
    Build feature matrix for NDVI prediction.
    Features: P_t, P_t-1, P_t-2, T2m_t, T2m_t-1, month_sin, month_cos, year_norm
    Target: ndvi
    """
    features = pd.DataFrame(index=df_zone.index)

    # Precipitation (current + 2 lags)
    if 'precip_mm' in df_zone.columns:
        features['P_t']   = df_zone['precip_mm']
        features['P_t1']  = df_zone['precip_mm'].shift(1)
        features['P_t2']  = df_zone['precip_mm'].shift(2)
        features['P_roll3'] = df_zone['precip_mm'].rolling(3, min_periods=1).mean()
    else:
        for c in ['P_t','P_t1','P_t2','P_roll3']:
            features[c] = 0.0

    # Temperature
    if 't2m_c' in df_zone.columns:
        features['T2m_t']  = df_zone['t2m_c']
        features['T2m_t1'] = df_zone['t2m_c'].shift(1)
    else:
        features['T2m_t']  = 20.0
        features['T2m_t1'] = 20.0

    # Seasonal
    features['month_sin'] = df_zone['month_sin'] if 'month_sin' in df_zone.columns else \
        np.sin(2*np.pi*df_zone.index.month/12)
    features['month_cos'] = df_zone['month_cos'] if 'month_cos' in df_zone.columns else \
        np.cos(2*np.pi*df_zone.index.month/12)
    features['year_norm'] = df_zone['year_norm'] if 'year_norm' in df_zone.columns else \
        (df_zone.index.year - 2002) / 22.0

    # SPI if available
    if 'spi3' in df_zone.columns:
        features['spi3'] = df_zone['spi3']

    target = df_zone['ndvi'] if 'ndvi' in df_zone.columns else None
    return features.dropna(), target


# ==============================================================================
# TRAINING
# ==============================================================================

def train_ndvi_emulator(zone, df_zone):
    """Train RF NDVI emulator for one zone."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    X, y = build_ndvi_features(df_zone)
    if y is None:
        log.warning(f"  Zone {zone}: no NDVI target — skipping")
        return None, None, None

    # Align
    common_idx = X.index.intersection(y.dropna().index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Train/val split
    train_mask = X.index <= TRAIN_END
    val_mask   = X.index >  TRAIN_END

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    if len(X_train) < 20:
        log.warning(f"  Zone {zone}: insufficient training data ({len(X_train)} rows)")
        return None, None, None

    log.info(f"  Training RF: {len(X_train)} train, {len(X_val)} val samples")

    model = RandomForestRegressor(
        n_estimators=N_EST,
        max_depth=MAX_D,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train.values, y_train.values)

    # Training metrics
    y_pred_train = model.predict(X_train.values)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # Validation metrics
    metrics = {'zone': zone, 'n_train': len(X_train), 'n_val': len(X_val)}
    if len(X_val) > 0:
        y_pred_val = model.predict(X_val.values)
        r2_val   = r2_score(y_val, y_pred_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
        metrics.update({
            'r2_train': r2_train, 'rmse_train': rmse_train,
            'r2_val':   r2_val,   'rmse_val':   rmse_val,
        })
        log.info(f"  Val R2={r2_val:.3f}, RMSE={rmse_val:.4f}")
    else:
        metrics.update({'r2_train': r2_train, 'rmse_train': rmse_train,
                        'r2_val': np.nan, 'rmse_val': np.nan})

    # Feature importance
    feat_names = X.columns.tolist()
    importances = dict(zip(feat_names, model.feature_importances_))
    log.info(f"  Top features: " +
             ", ".join(f"{k}={v:.3f}" for k,v in
                       sorted(importances.items(), key=lambda x: -x[1])[:4]))

    return model, X.columns.tolist(), metrics


# ==============================================================================
# HISTORICAL RECONSTRUCTION + FUTURE PROJECTION
# ==============================================================================

def predict_ndvi_historical(model, feat_cols, df_zone, zone):
    """Apply emulator to reconstruct historical NDVI."""
    X, _ = build_ndvi_features(df_zone)
    X = X[[c for c in feat_cols if c in X.columns]]
    # Fill missing cols with 0
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols]

    y_pred = pd.Series(model.predict(X.values), index=X.index, name=f'ndvi_emulated_{zone}')
    return y_pred


def load_cmip6_zone_means(zone):
    """
    Load CMIP6 P+T2m projections for a zone under SSP2-4.5 and SSP5-8.5.
    Returns dict with keys 'ssp245' and 'ssp585', each a DataFrame with P, T2m columns.
    Returns None if CMIP6 data not available.
    """
    cmip6_dir = Path(CFG['paths']['data']['cmip6'])
    if not cmip6_dir.exists():
        return None

    # Look for preprocessed CMIP6 files
    result = {}
    for ssp in ['ssp245', 'ssp585']:
        files = list(cmip6_dir.glob(f'*{ssp}*{zone}*.csv')) + \
                list(cmip6_dir.glob(f'*{zone}*{ssp}*.csv'))
        if files:
            df = pd.read_csv(files[0], parse_dates=[0], index_col=0)
            result[ssp] = df
            log.info(f"  CMIP6 {ssp} {zone}: {len(df)} months")

    return result if result else None


def make_future_features(ssp_df, zone, proj_start='2025-01-01', proj_end='2030-12-01'):
    """Build feature matrix for future projection from CMIP6 P+T2m."""
    idx = pd.date_range(proj_start, proj_end, freq='MS')
    features = pd.DataFrame(index=idx)

    # P from CMIP6 or synthetic trend
    if ssp_df is not None and 'P' in ssp_df.columns:
        P = ssp_df['P'].reindex(idx).ffill().bfill()
    elif ssp_df is not None and 'precip' in ssp_df.columns:
        P = ssp_df['precip'].reindex(idx).ffill().bfill()
    else:
        # Synthetic: slight drying trend
        m = idx.month
        t = np.arange(len(idx))
        base = {'north': 40, 'central': 20, 'south': 10}[zone]
        P = pd.Series(base * (1 - 0.005*t) + 0.3*np.random.randn(len(idx)), index=idx)
        P = P.clip(lower=0)

    # T2m from CMIP6 or warming trend
    if ssp_df is not None and 'T2m' in ssp_df.columns:
        T2m = ssp_df['T2m'].reindex(idx).ffill().bfill()
    elif ssp_df is not None and 'tas' in ssp_df.columns:
        T2m = ssp_df['tas'].reindex(idx).ffill().bfill()
        if T2m.mean() > 100:
            T2m = T2m - 273.15
    else:
        # Synthetic warming
        base_t = {'north': 18, 'central': 22, 'south': 26}[zone]
        m = idx.month
        t = np.arange(len(idx))
        T2m = pd.Series(base_t + 10*np.cos(2*np.pi*(m-7)/12) + 0.03*t, index=idx)

    features['P_t']     = P.values
    features['P_t1']    = P.shift(1).values
    features['P_t2']    = P.shift(2).values
    features['P_roll3'] = P.rolling(3, min_periods=1).mean().values
    features['T2m_t']   = T2m.values
    features['T2m_t1']  = T2m.shift(1).values
    features['month_sin'] = np.sin(2*np.pi*idx.month/12)
    features['month_cos'] = np.cos(2*np.pi*idx.month/12)
    features['year_norm'] = (idx.year - 2002) / 22.0
    features['spi3'] = 0.0  # neutral for future

    return features.ffill().bfill()


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_ndvi_emulator(zone, df_zone, y_emulated, y_pred_val=None, out_path=None):
    """Figure 05 — Observed vs emulated NDVI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Top: time series
    ax = axes[0]
    if 'ndvi' in df_zone.columns:
        ax.plot(df_zone.index, df_zone['ndvi'], 'k-', lw=1.2,
                label='MODIS observed', alpha=0.8)
    ax.plot(y_emulated.index, y_emulated.values, 'r-', lw=1.0,
            label='RF emulated', alpha=0.8)
    ax.axvline(TRAIN_END, color='gray', linestyle='--', lw=1, label='Train/Val split')
    ax.set_ylabel('NDVI')
    ax.set_title(f'NDVI Emulator — {zone.capitalize()} Zone', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom: scatter observed vs emulated
    ax2 = axes[1]
    if 'ndvi' in df_zone.columns:
        obs = df_zone['ndvi'].dropna()
        emu = y_emulated.reindex(obs.index).dropna()
        common = obs.index.intersection(emu.index)
        if len(common) > 5:
            ax2.scatter(obs[common], emu[common], alpha=0.4, s=15, color='steelblue')
            lim = [min(obs[common].min(), emu[common].min()) - 0.01,
                   max(obs[common].max(), emu[common].max()) + 0.01]
            ax2.plot(lim, lim, 'k--', lw=1)
            ax2.set_xlim(lim); ax2.set_ylim(lim)
            ax2.set_xlabel('NDVI observed')
            ax2.set_ylabel('NDVI emulated')

            from numpy.polynomial import polynomial as P_fit
            corr = np.corrcoef(obs[common], emu[common])[0,1]
            ax2.set_title(f'Scatter — r={corr:.3f}', fontsize=10)

    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(SEP)
    print('05_ndvi_emulator.py')
    print(SEP)

    # Load master features
    log.info('[STEP 1] Chargement features_master.csv ...')
    master_path = OUT_PROC / 'features_master.csv'
    if not master_path.exists():
        raise FileNotFoundError("features_master.csv not found. Run 04 first.")

    df_master = pd.read_csv(master_path, parse_dates=['time'], index_col='time')
    df_master.index = df_master.index.to_period('M').to_timestamp()
    log.info(f"  Loaded: {df_master.shape}, zones: {df_master['zone'].unique()}")

    # Train emulator per zone
    log.info('[STEP 2] Entrainement NDVI emulator par zone ...')
    models       = {}
    feat_cols_by = {}
    all_metrics  = []
    ndvi_hist    = {}

    for zone in ZONES:
        log.info(f'\n  -- Zone: {zone.upper()} --')
        df_z = df_master[df_master['zone'] == zone].copy()

        model, feat_cols, metrics = train_ndvi_emulator(zone, df_z)
        if model is None:
            continue

        models[zone]       = model
        feat_cols_by[zone] = feat_cols
        all_metrics.append(metrics)

        # Save model
        model_path = OUT_MOD / f'ndvi_emulator_{zone}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'feat_cols': feat_cols}, f)
        log.info(f'  [OK] Model: {model_path}')

        # Historical reconstruction
        y_hist = predict_ndvi_historical(model, feat_cols, df_z, zone)
        ndvi_hist[zone] = y_hist

        # Plot
        plot_ndvi_emulator(
            zone, df_z, y_hist,
            out_path=OUT_FIG / f'05_ndvi_emulator_{zone}.png'
        )
        log.info(f'  [OK] Figure: 05_ndvi_emulator_{zone}.png')

    # Save historical reconstruction
    log.info('\n[STEP 3] Sauvegarde reconstruction historique ...')
    if ndvi_hist:
        df_hist = pd.DataFrame(ndvi_hist)
        df_hist.columns = [f'ndvi_emulated_{z}' for z in df_hist.columns]
        df_hist.to_csv(OUT_PROC / 'ndvi_emulated_historical.csv')
        log.info(f'  [OK] ndvi_emulated_historical.csv — {df_hist.shape}')

    # Future projections (2025-2030)
    log.info('[STEP 4] Projections futures NDVI 2025-2030 ...')
    for ssp in ['ssp245', 'ssp585']:
        ndvi_proj = {}
        for zone in models:
            cmip6 = load_cmip6_zone_means(zone)
            ssp_df = cmip6.get(ssp) if cmip6 else None
            fut_X  = make_future_features(ssp_df, zone)

            feat_cols = feat_cols_by[zone]
            for c in feat_cols:
                if c not in fut_X.columns:
                    fut_X[c] = 0.0
            fut_X = fut_X[feat_cols]

            y_fut = pd.Series(
                models[zone].predict(fut_X.ffill().bfill().values),
                index=fut_X.index,
                name=f'ndvi_{zone}'
            )
            ndvi_proj[zone] = y_fut

        if ndvi_proj:
            df_proj = pd.DataFrame(ndvi_proj)
            df_proj.columns = [f'ndvi_emulated_{z}' for z in df_proj.columns]
            out_proj = OUT_PROC / f'ndvi_emulated_{ssp}.csv'
            df_proj.to_csv(out_proj)
            log.info(f'  [OK] {out_proj} — {df_proj.shape}')

    # Save metrics
    log.info('[STEP 5] Sauvegarde metriques ...')
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(OUT_RES / 'ndvi_emulator_metrics.csv', index=False)

        print('\n' + SEP)
        print('NDVI EMULATOR METRICS')
        print(SEP)
        print(metrics_df.to_string(index=False, float_format='{:.4f}'.format))

    # Summary
    print('\n' + SEP)
    print('RESUME 05_ndvi_emulator.py')
    print(SEP)
    print(f'  Zones trained : {list(models.keys())}')
    print(f'  Train end     : {TRAIN_END.date()}')
    print(f'  Val start     : {VAL_START.date()}')
    print(f'  Projections   : 2025-2030 (SSP2-4.5, SSP5-8.5)')
    print()
    print('Outputs:')
    print(f'  {OUT_PROC}/ndvi_emulated_historical.csv')
    print(f'  {OUT_PROC}/ndvi_emulated_ssp245.csv')
    print(f'  {OUT_PROC}/ndvi_emulated_ssp585.csv')
    print(f'  {OUT_RES}/ndvi_emulator_metrics.csv')
    for z in ZONES:
        print(f'  {OUT_FIG}/05_ndvi_emulator_{z}.png')
    print()
    print('[DONE] Pret pour 06_trend_analysis.py')
    print(SEP)


if __name__ == '__main__':
    main()
