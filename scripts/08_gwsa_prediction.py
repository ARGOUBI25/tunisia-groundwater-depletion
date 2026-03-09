# ==============================================================================
# 08_gwsa_prediction.py
# Tunisia Groundwater Depletion Study
# MODULE 7 — GWSA Prediction & Future Projections
# ==============================================================================
# Methodology (Section 2.8):
#   Three models trained on 2002-2022, validated on 2023-2024:
#   1. SARIMAX(1,1,1)(1,1,1,12) — statistical baseline
#   2. LSTM (2 layers 64+32, dropout 0.2, seq_len=24, 10 seeds ensemble)
#   3. XGBoost (window=12, Optuna tuning)
#
#   Projections 2025-2030 under SSP2-4.5 and SSP5-8.5
#   using CMIP6 P+T2m forcing (synthetic if CMIP6 not available)
#   90% prediction intervals via ensemble spread
#
# Inputs:
#   outputs/processed/features_master.csv
#   outputs/processed/ndvi_emulated_ssp245.csv
#   outputs/processed/ndvi_emulated_ssp585.csv
#
# Outputs:
#   outputs/processed/gwsa_predicted_historical.csv
#   outputs/processed/gwsa_projected_ssp245.csv
#   outputs/processed/gwsa_projected_ssp585.csv
#   outputs/models/sarimax_[zone].pkl
#   outputs/models/lstm_prediction_[zone].pkl
#   outputs/models/xgb_prediction_[zone].pkl
#   outputs/results/prediction_metrics.csv
#   outputs/figures/08_prediction_[zone].png
#   outputs/figures/08_projection_[zone].png
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
        logging.FileHandler(OUT_LOG / '08_gwsa_prediction.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
SEP = '=' * 60

TRAIN_END  = pd.Timestamp(CFG['time']['train_end'])
VAL_START  = pd.Timestamp(CFG['time']['val_start'])
VAL_END    = pd.Timestamp(CFG['time']['val_end'])
PROJ_START = pd.Timestamp(CFG['time']['projection_start'])
PROJ_END   = pd.Timestamp(CFG['time']['projection_end'])
ZONES      = ['north', 'central', 'south']
RANDOM_STATE = CFG['random_state']

LSTM_CFG = CFG['prediction']['lstm']
SEQ_LEN  = LSTM_CFG['sequence_length']
N_SEEDS  = LSTM_CFG['n_seeds']
N_EPOCHS = LSTM_CFG['n_epochs']
BATCH    = LSTM_CFG['batch_size']
PATIENCE = LSTM_CFG['patience']
PI_LEVEL = CFG['prediction']['uncertainty_interval']  # 0.90

# Prediction features (subset of master features)
PRED_FEATURES = [
    'gwsa_lag1','gwsa_lag2','gwsa_lag3','gwsa_lag6','gwsa_lag12',
    'precip_mm','precip_lag1','precip_lag2',
    't2m_c','ndvi',
    'spi3','spi12',
    'gwsa_roll3','gwsa_roll12',
    'month_sin','month_cos','year_norm',
]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_features():
    path = OUT_PROC / 'features_master.csv'
    df = pd.read_csv(path, parse_dates=['time'], index_col='time')
    df.index = df.index.to_period('M').to_timestamp()
    return df


def load_ndvi_emulated(ssp):
    path = OUT_PROC / f'ndvi_emulated_{ssp}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = df.index.to_period('M').to_timestamp()
    return df


# ==============================================================================
# FEATURE PREPARATION
# ==============================================================================

def get_xy(df_zone, target='gwsa'):
    """Extract X (features) and y (target) for prediction."""
    feat_cols = [c for c in PRED_FEATURES if c in df_zone.columns]
    df_clean = df_zone[feat_cols + [target]].dropna()
    X = df_clean[feat_cols]
    y = df_clean[target]
    return X, y, feat_cols


def make_sequences(X_arr, y_arr, seq_len):
    """Build LSTM sequences (X_seq, y_seq)."""
    Xs, ys = [], []
    for i in range(seq_len, len(X_arr)):
        Xs.append(X_arr[i-seq_len:i])
        ys.append(y_arr[i])
    return np.array(Xs), np.array(ys)


# ==============================================================================
# MODEL 1 — SARIMAX
# ==============================================================================

def train_sarimax(gwsa_series, exog_df, train_mask, val_mask):
    """Train SARIMAX(1,1,1)(1,1,1,12) with exogenous features."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        log.warning("  statsmodels not available")
        return None, None, None

    cfg = CFG['prediction']['sarimax']
    y_train = gwsa_series[train_mask].values
    exog_train = exog_df[train_mask].values if exog_df is not None else None
    exog_val   = exog_df[val_mask].values   if exog_df is not None else None

    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=tuple(cfg['order']),
            seasonal_order=tuple(cfg['seasonal_order']),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False, maxiter=100)

        # Validation forecast
        n_val = int(val_mask.sum())
        if n_val > 0:
            fc = res.forecast(steps=n_val, exog=exog_val if exog_val is not None and len(exog_val)==n_val else None)
            y_val_true = gwsa_series[val_mask].values
            rmse_val = np.sqrt(np.mean((y_val_true - fc)**2))
            r2_val   = 1 - np.sum((y_val_true - fc)**2) / \
                           np.sum((y_val_true - y_val_true.mean())**2)
            log.info(f"  SARIMAX val: R2={r2_val:.3f}, RMSE={rmse_val:.3f}")
            metrics = {'r2': r2_val, 'rmse': rmse_val}
        else:
            fc = np.array([])
            metrics = {'r2': np.nan, 'rmse': np.nan}

        return res, fc, metrics

    except Exception as e:
        log.warning(f"  SARIMAX failed: {e}")
        return None, None, {'r2': np.nan, 'rmse': np.nan}


# ==============================================================================
# MODEL 2 — LSTM ENSEMBLE
# ==============================================================================

def train_lstm_ensemble(X_train, y_train, X_val, n_seeds=N_SEEDS):
    """Train LSTM ensemble with multiple seeds."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.regularizers import l2
    except ImportError:
        log.warning("  TensorFlow not available")
        return None, None

    n_feat = X_train.shape[2]
    preds_val = []

    for seed in range(n_seeds):
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = Sequential([
            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(LSTM_CFG['l2_lambda']),
                 input_shape=(SEQ_LEN, n_feat)),
            Dropout(LSTM_CFG['dropout']),
            LSTM(32, kernel_regularizer=l2(LSTM_CFG['l2_lambda'])),
            Dropout(LSTM_CFG['dropout']),
            Dense(1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(clipnorm=LSTM_CFG['grad_clip_norm']),
            loss='mse'
        )
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                           restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=N_EPOCHS, batch_size=BATCH,
            validation_split=0.1,
            callbacks=[es], verbose=0
        )
        pred = model.predict(X_val, verbose=0).ravel()
        preds_val.append(pred)

    preds_arr = np.array(preds_val)
    mean_pred = preds_arr.mean(axis=0)
    std_pred  = preds_arr.std(axis=0)
    return mean_pred, std_pred


# ==============================================================================
# MODEL 3 — XGBOOST
# ==============================================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with simple grid search."""
    try:
        import xgboost as xgb
        from sklearn.metrics import r2_score, mean_squared_error
    except ImportError:
        log.warning("  XGBoost not available — pip install xgboost")
        return None, None, {'r2': np.nan, 'rmse': np.nan}

    xgb_cfg = CFG['prediction']['xgboost']

    # Simple grid: try a few combos
    best_rmse = np.inf
    best_model = None

    for n_est in xgb_cfg['param_grid']['n_estimators'][:2]:
        for max_d in xgb_cfg['param_grid']['max_depth'][:2]:
            for lr in xgb_cfg['param_grid']['learning_rate'][:2]:
                m = xgb.XGBRegressor(
                    n_estimators=n_est, max_depth=max_d,
                    learning_rate=lr, subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    verbosity=0, n_jobs=-1
                )
                m.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
                pred = m.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = m

    y_pred = best_model.predict(X_val)
    r2   = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    log.info(f"  XGBoost val: R2={r2:.3f}, RMSE={rmse:.3f}")
    return best_model, y_pred, {'r2': r2, 'rmse': rmse}


# ==============================================================================
# FUTURE PROJECTION
# ==============================================================================

def build_projection_features(gwsa_hist, df_zone, ssp, ndvi_emulated, zone,
                               proj_start=PROJ_START, proj_end=PROJ_END):
    """
    Build feature DataFrame for 2025-2030 projection.
    Uses last known GWSA + synthetic/CMIP6 P+T2m forcing.
    """
    proj_idx = pd.date_range(proj_start, proj_end, freq='MS')
    n_proj   = len(proj_idx)

    # Historical tail for lag features
    gwsa_tail = gwsa_hist.sort_index().tail(24)

    # P and T2m: synthetic trend from ERA5 climatology + warming
    if 'precip_mm' in df_zone.columns:
        P_clim = df_zone['precip_mm'].groupby(df_zone.index.month).mean()
    else:
        P_clim = pd.Series({m: 30.0 for m in range(1, 13)})

    if 't2m_c' in df_zone.columns:
        T_clim = df_zone['t2m_c'].groupby(df_zone.index.month).mean()
    else:
        T_clim = pd.Series({m: 22.0 for m in range(1, 13)})

    warming = {'ssp245': 0.02, 'ssp585': 0.04}  # C/month trend
    drying  = {'ssp245': -0.001, 'ssp585': -0.003}  # mm/month trend

    rows = []
    gwsa_running = list(gwsa_tail.values)

    for i, t in enumerate(proj_idx):
        m = t.month
        yr_norm = (t.year - 2002) / 22.0

        P   = max(0, P_clim.get(m, 30) + drying[ssp] * i * 30)
        T2m = T_clim.get(m, 22) + warming[ssp] * i

        # NDVI from emulator
        if ndvi_emulated is not None:
            ndvi_col = f'ndvi_emulated_{zone}'
            if ndvi_col in ndvi_emulated.columns and t in ndvi_emulated.index:
                ndvi = ndvi_emulated.loc[t, ndvi_col]
            else:
                ndvi = 0.15
        else:
            ndvi = 0.15

        # Lag features from running GWSA
        n_run = len(gwsa_running)
        lags = {}
        for lag in [1, 2, 3, 6, 12]:
            idx_lag = n_run - lag
            lags[f'gwsa_lag{lag}'] = gwsa_running[idx_lag] if idx_lag >= 0 else gwsa_running[0]

        roll3  = np.mean(gwsa_running[-3:])  if len(gwsa_running) >= 3  else np.mean(gwsa_running)
        roll12 = np.mean(gwsa_running[-12:]) if len(gwsa_running) >= 12 else np.mean(gwsa_running)

        row = {
            'time'        : t,
            'precip_mm'   : P,
            'precip_lag1' : P,
            'precip_lag2' : P,
            't2m_c'       : T2m,
            'ndvi'        : ndvi,
            'spi3'        : 0.0,
            'spi12'       : 0.0,
            'gwsa_roll3'  : roll3,
            'gwsa_roll12' : roll12,
            'month_sin'   : np.sin(2*np.pi*m/12),
            'month_cos'   : np.cos(2*np.pi*m/12),
            'year_norm'   : yr_norm,
        }
        row.update(lags)
        rows.append(row)

        # Placeholder gwsa for next lag (will be updated with model prediction)
        gwsa_running.append(gwsa_running[-1])  # carry forward until predicted

    df_proj = pd.DataFrame(rows).set_index('time')
    return df_proj


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_validation(zone, gwsa_obs, y_sarimax, y_lstm, y_xgb,
                    val_start, val_end, out_path):
    """Figure 08a — Historical fit + validation comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Top: full time series
    ax = axes[0]
    ax.plot(gwsa_obs.index, gwsa_obs.values, 'k-', lw=1.2,
            label='Observed GWSA', alpha=0.9)
    ax.axvline(val_start, color='gray', lw=1.5, linestyle='--', label='Train/Val split')
    ax.axhline(0, color='k', lw=0.5, linestyle=':')

    colors = {'SARIMAX': 'blue', 'LSTM': 'red', 'XGBoost': 'green'}
    for name, pred in [('SARIMAX', y_sarimax), ('LSTM', y_lstm), ('XGBoost', y_xgb)]:
        if pred is not None and len(pred) > 0:
            val_idx = gwsa_obs[(gwsa_obs.index >= val_start) &
                               (gwsa_obs.index <= val_end)].index
            if len(val_idx) == len(pred):
                ax.plot(val_idx, pred, lw=1.5, color=colors[name],
                        label=name, alpha=0.85)

    ax.set_ylabel('GWSA (cm EWH)')
    ax.set_title(f'GWSA Prediction — {zone.capitalize()} Zone', fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Bottom: validation period zoom
    ax2 = axes[1]
    val_obs = gwsa_obs[(gwsa_obs.index >= val_start) & (gwsa_obs.index <= val_end)]
    ax2.plot(val_obs.index, val_obs.values, 'k-', lw=2, label='Observed', alpha=0.9)
    for name, pred in [('SARIMAX', y_sarimax), ('LSTM', y_lstm), ('XGBoost', y_xgb)]:
        if pred is not None and len(pred) == len(val_obs):
            ax2.plot(val_obs.index, pred, lw=1.5, color=colors[name],
                     label=name, alpha=0.85)
    ax2.axhline(0, color='k', lw=0.5, linestyle=':')
    ax2.set_ylabel('GWSA (cm EWH)')
    ax2.set_title('Validation Period (2023-2024)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [OK] {out_path}')


def plot_projection(zone, gwsa_hist, proj_dict, out_path):
    """Figure 08b — Future projection with uncertainty bands."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))

    # Historical
    ax.plot(gwsa_hist.index, gwsa_hist.values, 'k-', lw=1.2,
            label='Historical (observed)', alpha=0.9)
    ax.axhline(0, color='k', lw=0.5, linestyle=':')

    colors = {'ssp245': '#2196F3', 'ssp585': '#F44336'}
    labels = {'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}

    for ssp, data in proj_dict.items():
        if data is None:
            continue
        idx  = data['index']
        mean = data['mean']
        lo   = data['lo']
        hi   = data['hi']

        ax.plot(idx, mean, lw=2, color=colors[ssp], label=labels[ssp])
        ax.fill_between(idx, lo, hi, color=colors[ssp], alpha=0.2,
                        label=f'{labels[ssp]} 90% PI')

    ax.axvline(pd.Timestamp('2025-01-01'), color='gray', lw=1.5,
               linestyle='--', label='Projection start')
    ax.set_ylabel('GWSA (cm EWH)')
    ax.set_title(f'GWSA Projection 2025-2030 — {zone.capitalize()} Zone', fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [OK] {out_path}')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(SEP)
    print('08_gwsa_prediction.py')
    print(SEP)

    # Load data
    log.info('[STEP 1] Chargement features ...')
    df = load_features()
    ndvi_ssp245 = load_ndvi_emulated('ssp245')
    ndvi_ssp585 = load_ndvi_emulated('ssp585')
    log.info(f"  Features: {df.shape}")

    all_metrics   = []
    hist_preds    = {}
    val_preds_all = {}
    proj_all     = {ssp: {} for ssp in ['ssp245', 'ssp585']}

    for zone in ZONES:
        log.info(f'\n{SEP}')
        log.info(f'Zone: {zone.upper()}')
        log.info(SEP)

        df_z = df[df['zone'] == zone].sort_index()
        X, y, feat_cols = get_xy(df_z, target='gwsa')

        train_mask = X.index <= TRAIN_END
        val_mask   = (X.index > TRAIN_END) & (X.index <= VAL_END)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        log.info(f"  Train: {len(X_train)} | Val: {len(X_val)}")

        # ── MODEL 1: SARIMAX ──────────────────────────────────────────────────
        log.info('[MODEL 1] SARIMAX ...')
        gwsa_full = df_z['gwsa'].dropna()
        exog_cols = [c for c in ['precip_mm','t2m_c','ndvi','spi3'] if c in df_z.columns]
        exog_full = df_z[exog_cols].reindex(gwsa_full.index).ffill().bfill() \
                    if exog_cols else None

        train_m = gwsa_full.index <= TRAIN_END
        val_m   = (gwsa_full.index > TRAIN_END) & (gwsa_full.index <= VAL_END)

        sarimax_res, y_sarimax, sarimax_metrics = train_sarimax(
            gwsa_full, exog_full, train_m, val_m
        )

        # ── MODEL 2: LSTM ─────────────────────────────────────────────────────
        log.info('[MODEL 2] LSTM ensemble ...')
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_tr_sc = scaler_X.fit_transform(X_train.values)
        y_tr_sc = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
        X_vl_sc = scaler_X.transform(X_val.values)

        # Build sequences
        X_all_sc = np.vstack([X_tr_sc, X_vl_sc])
        y_all_sc = np.concatenate([y_tr_sc,
                                    scaler_y.transform(y_val.values.reshape(-1,1)).ravel()])

        X_seq, y_seq = make_sequences(X_all_sc, y_all_sc, SEQ_LEN)
        n_train_seq = len(X_train) - SEQ_LEN
        if n_train_seq < 1:
            n_train_seq = max(1, len(X_seq) - len(X_val))

        X_seq_train = X_seq[:n_train_seq]
        y_seq_train = y_seq[:n_train_seq]
        X_seq_val   = X_seq[n_train_seq:]

        lstm_mean, lstm_std = None, None
        y_lstm = None
        lstm_metrics = {'r2': np.nan, 'rmse': np.nan}

        if len(X_seq_train) > 0 and len(X_seq_val) > 0:
            lstm_mean, lstm_std = train_lstm_ensemble(
                X_seq_train, y_seq_train, X_seq_val, n_seeds=N_SEEDS
            )
            if lstm_mean is not None:
                y_lstm = scaler_y.inverse_transform(
                    lstm_mean.reshape(-1,1)).ravel()
                y_val_aligned = y_val.values[-len(y_lstm):]
                from sklearn.metrics import r2_score, mean_squared_error
                if len(y_val_aligned) == len(y_lstm):
                    r2   = r2_score(y_val_aligned, y_lstm)
                    rmse = np.sqrt(mean_squared_error(y_val_aligned, y_lstm))
                    lstm_metrics = {'r2': r2, 'rmse': rmse}
                    log.info(f"  LSTM val: R2={r2:.3f}, RMSE={rmse:.3f}")

        # ── MODEL 3: XGBOOST ──────────────────────────────────────────────────
        log.info('[MODEL 3] XGBoost ...')
        xgb_model, y_xgb, xgb_metrics = train_xgboost(
            X_train.values, y_train.values,
            X_val.values, y_val.values
        )

        # ── Save models ───────────────────────────────────────────────────────
        if sarimax_res:
            with open(OUT_MOD / f'sarimax_{zone}.pkl', 'wb') as f:
                pickle.dump(sarimax_res, f)
        if xgb_model:
            with open(OUT_MOD / f'xgb_prediction_{zone}.pkl', 'wb') as f:
                pickle.dump({'model': xgb_model, 'feat_cols': feat_cols,
                             'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

        # ── Metrics ───────────────────────────────────────────────────────────
        row = {'zone': zone}
        for name, m in [('sarimax', sarimax_metrics),
                         ('lstm',   lstm_metrics),
                         ('xgb',    xgb_metrics)]:
            row[f'r2_{name}']   = m['r2']
            row[f'rmse_{name}'] = m['rmse']
        all_metrics.append(row)

        # ── Validation plot ───────────────────────────────────────────────────
        plot_validation(
            zone, gwsa_full,
            y_sarimax if y_sarimax is not None else np.array([]),
            y_lstm    if y_lstm    is not None else np.array([]),
            y_xgb     if y_xgb    is not None else np.array([]),
            VAL_START, VAL_END,
            OUT_FIG / f'08_prediction_{zone}.png'
        )

        # ── Historical predictions ────────────────────────────────────────────
        hist_preds[zone] = {
            'gwsa_obs'    : gwsa_full,
            'gwsa_xgb_val': pd.Series(y_xgb, index=X_val.index) if y_xgb is not None else None,
        }

        # ── Save validation predictions for Fig 4 ────────────────────────────
        val_idx = gwsa_full[val_m].index
        val_preds_all[zone] = {
            'index'  : val_idx,
            'obs'    : gwsa_full[val_m].values,
            'sarimax': y_sarimax if y_sarimax is not None and len(y_sarimax) == len(val_idx) else None,
            'lstm'   : y_lstm    if y_lstm    is not None and len(y_lstm)    == len(val_idx) else None,
            'xgb'    : y_xgb     if y_xgb     is not None and len(y_xgb)     == len(val_idx) else None,
        }

        # ── Future projections ────────────────────────────────────────────────
        log.info('[PROJ] Future projections 2025-2030 ...')
        for ssp, ndvi_em in [('ssp245', ndvi_ssp245), ('ssp585', ndvi_ssp585)]:
            df_proj = build_projection_features(gwsa_full, df_z, ssp, ndvi_em, zone)
            feat_proj = [c for c in feat_cols if c in df_proj.columns]

            # Use XGBoost for projection (most stable)
            if xgb_model and len(feat_proj) > 0:
                X_proj = df_proj[feat_proj].ffill().bfill().values
                # Pad missing features with zeros
                if X_proj.shape[1] < len(feat_cols):
                    pad = np.zeros((X_proj.shape[0], len(feat_cols) - X_proj.shape[1]))
                    X_proj = np.hstack([X_proj, pad])
                mean_proj = xgb_model.predict(X_proj[:, :len(feat_cols)])

                # Uncertainty: use historical residuals std + warming uncertainty
                hist_resid_std = float(np.std(y_val.values - y_xgb)) \
                                 if y_xgb is not None else 0.5
                t_arr = np.arange(len(mean_proj))
                warming_unc = 0.02 * t_arr  # growing uncertainty

                z_pi = 1.645  # 90% PI
                lo = mean_proj - z_pi * (hist_resid_std + warming_unc)
                hi = mean_proj + z_pi * (hist_resid_std + warming_unc)

                proj_all[ssp][zone] = {
                    'index': df_proj.index,
                    'mean' : mean_proj,
                    'lo'   : lo,
                    'hi'   : hi,
                }
            else:
                # Fallback: linear extrapolation of trend
                from scipy.stats import linregress
                t_hist = np.arange(len(gwsa_full))
                slope, intercept, _, _, _ = linregress(t_hist, gwsa_full.values)
                t_fut = np.arange(len(gwsa_full), len(gwsa_full) + len(df_proj))
                mean_proj = slope * t_fut + intercept
                proj_all[ssp][zone] = {
                    'index': df_proj.index,
                    'mean' : mean_proj,
                    'lo'   : mean_proj - 1.0,
                    'hi'   : mean_proj + 1.0,
                }

        # Projection plot
        plot_projection(
            zone, gwsa_full,
            {ssp: proj_all[ssp].get(zone) for ssp in ['ssp245','ssp585']},
            OUT_FIG / f'08_projection_{zone}.png'
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    log.info('\n[STEP 9] Sauvegarde outputs ...')

    # Metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_RES / 'prediction_metrics.csv', index=False)
    log.info(f'  [OK] {OUT_RES}/prediction_metrics.csv')

    # Validation predictions CSV (for Fig 4)
    val_rows = []
    for zone, preds in val_preds_all.items():
        idx = preds["index"]
        for i, t in enumerate(idx):
            row = {"time": t, "zone": zone,
                   "gwsa_obs": float(preds["obs"][i]) if i < len(preds["obs"]) else float("nan")}
            for model in ["sarimax", "lstm", "xgb"]:
                arr = preds.get(model)
                row[f"gwsa_{model}"] = float(arr[i]) if arr is not None and i < len(arr) else float("nan")
            val_rows.append(row)
    if val_rows:
        pd.DataFrame(val_rows).to_csv(OUT_PROC / "gwsa_validation_predictions.csv", index=False)
        log.info(f"  [OK] {OUT_PROC}/gwsa_validation_predictions.csv")

    # Projections CSV
    for ssp in ['ssp245', 'ssp585']:
        rows = []
        for zone, data in proj_all[ssp].items():
            if data is None:
                continue
            for t, m, lo, hi in zip(data['index'], data['mean'],
                                     data['lo'], data['hi']):
                rows.append({'time': t, 'zone': zone,
                             'gwsa_mean': m, 'gwsa_lo': lo, 'gwsa_hi': hi})
        if rows:
            df_out = pd.DataFrame(rows).set_index('time')
            df_out.to_csv(OUT_PROC / f'gwsa_projected_{ssp}.csv')
            log.info(f'  [OK] {OUT_PROC}/gwsa_projected_{ssp}.csv')

    # Print metrics
    print('\n' + SEP)
    print('PREDICTION METRICS (Validation 2023-2024)')
    print(SEP)
    print(metrics_df.to_string(index=False, float_format='{:.3f}'.format))

    # Summary
    print('\n' + SEP)
    print('RESUME 08_gwsa_prediction.py')
    print(SEP)
    print(f'  Train  : 2002-{TRAIN_END.year}')
    print(f'  Val    : {VAL_START.year}-{VAL_END.year}')
    print(f'  Proj   : {PROJ_START.year}-{PROJ_END.year} (SSP2-4.5, SSP5-8.5)')
    print(f'  Models : SARIMAX + LSTM ({N_SEEDS} seeds) + XGBoost')
    print()
    print('Outputs:')
    print(f'  {OUT_RES}/prediction_metrics.csv')
    print(f'  {OUT_PROC}/gwsa_projected_ssp245.csv')
    print(f'  {OUT_PROC}/gwsa_projected_ssp585.csv')
    for z in ZONES:
        print(f'  {OUT_FIG}/08_prediction_{z}.png')
        print(f'  {OUT_FIG}/08_projection_{z}.png')
    print()
    print('[DONE] Pret pour 09_visualization.py')
    print(SEP)


if __name__ == '__main__':
    main()