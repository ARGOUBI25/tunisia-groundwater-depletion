# ==============================================================================
# 09_visualization.py
# Tunisia Groundwater Depletion Study
# MODULE 9 — Publication-ready figures (uniform style, 300 DPI)
# ==============================================================================
# Regenerates all key figures from saved CSV/NetCDF outputs with:
#   - Uniform font (DejaVu Sans, 10pt body / 11pt titles)
#   - Consistent color palette
#   - 300 DPI, tight bbox, no white borders
#   - Journal-ready panel labels (a, b, c ...)
#
# Figures produced:
#   Fig 1  — GWSA time series + gap fill (3 zones, 1 panel each)
#   Fig 2  — Downscaling map + scatter (4-panel)
#   Fig 3  — MMK trend summary (3x4 grid)
#   Fig 4  — Water stress time series + cluster scatter (2-panel)
#   Fig 5  — Prediction validation (3 zones)
#   Fig 6  — GWSA projections SSP2-4.5 vs SSP5-8.5 (3 zones)
#
# Inputs  : outputs/processed/*.csv, outputs/results/*.csv
# Outputs : outputs/figures/pub_fig[N]_*.png  (300 DPI)
# ==============================================================================

import os
import sys
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
import io

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

with open('config.yaml', 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

OUT_PROC = Path(CFG['paths']['outputs']['processed'])
OUT_RES  = Path(CFG['paths']['outputs']['results'])
OUT_FIG  = Path(CFG['paths']['outputs']['figures'])
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEP = '=' * 60

# ==============================================================================
# GLOBAL STYLE
# ==============================================================================

STYLE = {
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
    'figure.dpi'        : 150,       # screen preview
    'savefig.dpi'       : 300,       # publication output
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.25,
    'grid.linewidth'    : 0.5,
    'lines.linewidth'   : 1.4,
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : 'white',
}
plt.rcParams.update(STYLE)

# Colour palette — consistent across all figures
PAL = {
    'north'    : '#1a6faf',   # deep blue
    'central'  : '#e07b00',   # amber
    'south'    : '#2e9e52',   # green
    'obs'      : '#222222',   # near-black
    'sarimax'  : '#5b8dd9',   # light blue
    'lstm'     : '#e05c5c',   # red
    'xgb'      : '#52b788',   # teal
    'ssp245'   : '#2196F3',   # blue
    'ssp585'   : '#F44336',   # red
    'low'      : '#2ecc71',
    'moderate' : '#f39c12',
    'high'     : '#e74c3c',
    'critical' : '#8e44ad',
    'gap'      : '#dddddd',
}

ZONES       = ['north', 'central', 'south']
ZONE_LABELS = {'north': 'Northern Zone', 'central': 'Central Zone',
               'south': 'Southern Zone'}
STRESS_ORDER = ['low', 'moderate', 'high', 'critical']

DPI_SAVE = 300


def save(fig, name):
    path = OUT_FIG / f'pub_{name}.png'
    fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] {path}')


def panel_label(ax, letter, x=-0.10, y=1.05):
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')


# ==============================================================================
# FIG 1 — GWSA TIME SERIES + GAP FILL
# ==============================================================================

def fig1_gwsa_timeseries():
    """3-zone GWSA time series with gap-filled period highlighted.
    Handles wide format: columns gwsa_north, gwsa_central, gwsa_south."""
    path = OUT_PROC / 'gwsa_zones_gap_filled.csv'
    if not path.exists():
        print('  [SKIP] gwsa_zones_gap_filled.csv not found')
        return

    df = pd.read_csv(path, encoding='utf-8-sig')
    df.rename(columns={df.columns[0]: 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Load original (pre-gap-fill) for overlay
    orig_path = OUT_PROC / 'gwsa_zones_monthly.csv'
    df_orig = None
    if orig_path.exists():
        df_orig = pd.read_csv(orig_path, encoding='utf-8-sig')
        df_orig.rename(columns={df_orig.columns[0]: 'time'}, inplace=True)
        df_orig['time'] = pd.to_datetime(df_orig['time'])
        df_orig = df_orig.sort_values('time').reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    gap_start = pd.Timestamp('2017-07-01')
    gap_end   = pd.Timestamp('2018-06-01')

    letters = ['a', 'b', 'c']
    for ax, zone, letter in zip(axes, ZONES, letters):
        col = f'gwsa_{zone}'           # wide format column name
        if col not in df.columns:
            print(f'  [WARN] column {col} not found, skipping')
            continue

        # Clip to end of 2024 - CSV extends into 2025 causing spike
        df_clip = df[df['time'] <= pd.Timestamp('2024-12-31')]
        y = df_clip[col].values
        t = df_clip['time']

        ax.axvspan(gap_start, gap_end, color=PAL['gap'], alpha=0.5, zorder=0)
        ax.axhline(0, color='k', lw=0.6, linestyle=':', zorder=1)
        ax.fill_between(t, y, 0, where=y < 0,
                        color=PAL[zone], alpha=0.20, zorder=2)
        ax.plot(t, y, color=PAL[zone], lw=1.4,
                label='GWSA (gap-filled)', zorder=3)

        # Original overlay
        orig_col = f'gwsa_{zone}' if df_orig is not None and f'gwsa_{zone}' in df_orig.columns else None
        if orig_col:
            ax.plot(df_orig['time'], df_orig[orig_col],
                    color=PAL['obs'], lw=0.8, alpha=0.5,
                    linestyle=':', label='Original (pre-fill)', zorder=4)

        ax.set_ylabel('GWSA (cm EWH)', fontsize=9)
        _ztitle = {'north':'Northern Zone','central':'Central Zone','south':'Southern Zone'}
        ax.set_title(_ztitle.get(zone, ZONE_LABELS[zone]), fontsize=10, loc='left', pad=3)
        panel_label(ax, letter)

        handles = [
            Line2D([0],[0], color=PAL[zone], lw=1.4, label='GWSA (gap-filled)'),
            mpatches.Patch(color=PAL['gap'], alpha=0.5,
                           label='Gap period (Jul 2017 - May 2018)'),
        ]
        if orig_col:
            handles.append(Line2D([0],[0], color=PAL['obs'], lw=0.8,
                                   linestyle=':', alpha=0.5,
                                   label='Original (pre-fill)'))
        ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.7)

    axes[-1].set_xlabel('Date')
    fig.suptitle('GRACE/GRACE-FO Groundwater Storage Anomaly - Tunisia (2002-2024)',
                 fontsize=11, y=1.01)
    save(fig, 'fig1_gwsa_timeseries')

# ==============================================================================
# FIG 2 — TREND ANALYSIS SUMMARY (MMK)
# ==============================================================================

def fig2_trend_summary():
    """4-panel trend figure: GWSA, Precip, T2m, NDVI per zone."""
    path = OUT_RES / 'trend_results.csv'
    if not path.exists():
        print('  [SKIP] trend_results.csv not found')
        return

    tr = pd.read_csv(path)

    # Also load time series for background
    feat_path = OUT_PROC / 'features_master.csv'
    if not feat_path.exists():
        print('  [SKIP] features_master.csv not found')
        return
    df = pd.read_csv(feat_path, parse_dates=['time'])
    df['time'] = pd.to_datetime(df['time'])

    variables = [
        ('gwsa',       'GWSA (cm EWH)',     'gwsa'),
        ('precip_mm',  'Precipitation (mm)', 'precip_mm'),
        ('t2m_c',      'Temperature (°C)',   't2m_c'),
        ('ndvi',       'NDVI',               'ndvi'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    letters = ['a', 'b', 'c', 'd']

    for ax, (var, ylabel, col), letter in zip(axes, variables, letters):
        for zone in ZONES:
            dz = df[df['zone'] == zone].sort_values('time')
            if col not in dz.columns:
                continue

            y = dz[col].values
            t = np.arange(len(y))

            ax.plot(dz['time'], y, color=PAL[zone], lw=1.0, alpha=0.6)

            # Sen's slope line
            row = tr[(tr['zone'] == zone) & (tr['variable'] == var)]
            if len(row) > 0:
                slope  = float(row['sen_slope_yr'].values[0]) / 12  # per month
                sig    = str(row['trend'].values[0])
                y_mean = np.nanmean(y)
                t_mid  = len(y) / 2
                y_fit  = y_mean + slope * (t - t_mid)

                lw_fit   = 2.0 if sig == 'decreasing' or sig == 'increasing' else 1.0
                ls_fit   = '-'  if sig != 'no trend' else '--'
                alpha_fit = 1.0 if sig != 'no trend' else 0.4

                ax.plot(dz['time'], y_fit, color=PAL[zone],
                        lw=lw_fit, linestyle=ls_fit, alpha=alpha_fit,
                        label=f'{ZONE_LABELS[zone]} ({row["sen_slope_yr"].values[0]:+.3f}/yr'
                              f'{" *" if sig != "no trend" else ""})')

        ax.axhline(0, color='k', lw=0.5, linestyle=':', alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('Date', fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7, loc='lower left')
        panel_label(ax, letter)

    fig.suptitle('Modified Mann-Kendall Trend Analysis — Tunisia (2002–2024)',
                 fontsize=11)
    fig.tight_layout()
    save(fig, 'fig2_trend_summary')


# ==============================================================================
# FIG 3 — WATER STRESS TIME SERIES
# ==============================================================================

def fig3_water_stress():
    """3-zone water stress time series + annual distribution bar chart."""
    path = OUT_PROC / 'water_stress_classified.csv'
    if not path.exists():
        print('  [SKIP] water_stress_classified.csv not found')
        return

    df = pd.read_csv(path, parse_dates=['time'])
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            height_ratios=[1, 1, 1, 1.4],
                            hspace=0.35, wspace=0.35)

    # Top 3 rows: stress time series per zone
    letters = ['a', 'b', 'c', 'd']
    stress_colors = {
        'low': PAL['low'], 'moderate': PAL['moderate'],
        'high': PAL['high'], 'critical': PAL['critical']
    }

    for i, zone in enumerate(ZONES):
        ax = fig.add_subplot(gs[i, :])
        dz = df[df['zone'] == zone].sort_values('time')

        if 'stress_label' not in dz.columns:
            continue

        for cls in STRESS_ORDER:
            mask = dz['stress_label'] == cls
            if mask.any():
                ax.fill_between(dz['time'], 0, 1,
                                where=mask.values,
                                color=stress_colors[cls],
                                alpha=0.75, step='mid',
                                label=cls.capitalize())

        # GWSA overlay
        if 'gwsa' in dz.columns:
            ax2 = ax.twinx()
            ax2.plot(dz['time'], dz['gwsa'], color='k',
                     lw=1.0, alpha=0.6, label='GWSA')
            ax2.axhline(0, color='k', lw=0.4, linestyle=':')
            ax2.set_ylabel('GWSA\n(cm EWH)', fontsize=8)
            ax2.tick_params(labelsize=8)

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(ZONE_LABELS[zone], fontsize=9, rotation=90,
                      labelpad=6)
        ax.set_xlabel('')
        panel_label(ax, letters[i])

        if i == 0:
            patches = [mpatches.Patch(color=stress_colors[c],
                                      label=c.capitalize(),
                                      alpha=0.75)
                       for c in STRESS_ORDER]
            ax.legend(handles=patches, loc='upper right',
                      fontsize=8, ncol=4, framealpha=0.8)

    # Bottom row: annual stacked bar chart (all zones pooled)
    ax_bar = fig.add_subplot(gs[3, :])
    counts = df.groupby(['year', 'stress_label']).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    bottom = np.zeros(len(pct))
    for cls in STRESS_ORDER:
        if cls in pct.columns:
            ax_bar.bar(pct.index, pct[cls], bottom=bottom,
                       color=stress_colors[cls], alpha=0.85,
                       label=cls.capitalize(), width=0.8)
            bottom += pct[cls].values

    ax_bar.set_xlabel('Year', fontsize=9)
    ax_bar.set_ylabel('% of zone-months', fontsize=9)
    ax_bar.set_title('Annual Water Stress Distribution (all zones)', fontsize=10)
    ax_bar.legend(fontsize=8, ncol=4, loc='upper left', framealpha=0.8)
    ax_bar.set_ylim(0, 105)
    panel_label(ax_bar, letters[3])

    fig.suptitle('Water Stress Classification — Tunisia Hydro-climatic Zones (2002–2024)',
                 fontsize=11, y=1.01)
    save(fig, 'fig3_water_stress')


# ==============================================================================
# FIG 4 — PREDICTION VALIDATION (3 zones)
# ==============================================================================

def fig4_prediction_validation():
    """Validation figure: observed vs SARIMAX / LSTM / XGBoost per zone."""
    feat_path = OUT_PROC / 'features_master.csv'
    met_path  = OUT_RES  / 'prediction_metrics.csv'
    val_path  = OUT_PROC / 'gwsa_validation_predictions.csv'
    if not feat_path.exists() or not met_path.exists():
        print('  [SKIP] prediction inputs not found')
        return

    df  = pd.read_csv(feat_path, parse_dates=['time'])
    df['time'] = pd.to_datetime(df['time'])
    met = pd.read_csv(met_path)
    df_val = None
    if val_path.exists():
        df_val = pd.read_csv(val_path, parse_dates=['time'])
        df_val['time'] = pd.to_datetime(df_val['time'])

    val_start = pd.Timestamp(CFG['time']['val_start'])
    val_end   = pd.Timestamp(CFG['time']['val_end'])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    fig.subplots_adjust(hspace=0.45)

    letters = ['a', 'b', 'c']
    for ax, zone, letter in zip(axes, ZONES, letters):
        dz = df[df['zone'] == zone].sort_values('time')
        if 'gwsa' not in dz.columns:
            continue

        # Full observed series
        ax.plot(dz['time'], dz['gwsa'], color=PAL['obs'],
                lw=1.2, label='Observed GWSA', zorder=5)
        ax.axvline(val_start, color='gray', lw=1.2,
                   linestyle='--', alpha=0.7, label='Train / Val split')
        ax.axhline(0, color='k', lw=0.4, linestyle=':', alpha=0.5)

        # Shade validation period
        ax.axvspan(val_start, val_end, color='#f0f0f0', alpha=0.6, zorder=0)

        # Metrics annotation
        row = met[met['zone'] == zone]
        if len(row) > 0:
            r = row.iloc[0]
            txt_lines = []
            for name, key_r2, key_rmse, col in [
                ('SARIMAX', 'r2_sarimax', 'rmse_sarimax', PAL['sarimax']),
                ('LSTM',    'r2_lstm',    'rmse_lstm',    PAL['lstm']),
                ('XGBoost', 'r2_xgb',    'rmse_xgb',     PAL['xgb']),
            ]:
                r2   = r.get(key_r2,   float('nan'))
                rmse = r.get(key_rmse, float('nan'))
                if not np.isnan(r2):
                    txt_lines.append(f'{name}: R²={r2:.3f}, RMSE={rmse:.3f}')

            ax.text(0.99, 0.97, '\n'.join(txt_lines),
                    transform=ax.transAxes,
                    fontsize=7.5, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8, edgecolor='#cccccc'))

        # ── Model prediction curves ──────────────────────────────────────────
        if df_val is not None:
            dv = df_val[df_val['zone'] == zone].sort_values('time')
            model_lines = [
                ('gwsa_sarimax', PAL['sarimax'], 'SARIMAX', '-'),
                ('gwsa_lstm',    PAL['lstm'],    'LSTM',    '--'),
                ('gwsa_xgb',     PAL['xgb'],     'XGBoost', '-.'),
            ]
            for col, color, label, ls in model_lines:
                if col in dv.columns and not dv[col].isna().all():
                    ax.plot(dv['time'], dv[col], color=color,
                            lw=1.8, linestyle=ls, label=label, zorder=6)

        ax.set_ylabel('GWSA (cm EWH)', fontsize=9)
        ax.set_title(ZONE_LABELS[zone], fontsize=10, loc='left', pad=3)
        ax.set_xlabel('Date', fontsize=9)

        # Legend
        handles = [
            Line2D([0],[0], color=PAL['obs'],    lw=1.2, label='Observed'),
            Line2D([0],[0], color='gray',         lw=1.2, ls='--', label='Train/Val split'),
            mpatches.Patch(color='#f0f0f0', alpha=0.8, label='Validation period'),
            Line2D([0],[0], color=PAL['sarimax'], lw=1.8, ls='-',  label='SARIMAX'),
            Line2D([0],[0], color=PAL['lstm'],    lw=1.8, ls='--', label='LSTM'),
            Line2D([0],[0], color=PAL['xgb'],     lw=1.8, ls='-.', label='XGBoost'),
        ]
        ax.legend(handles=handles, fontsize=8, loc='lower left',
                  framealpha=0.8, ncol=2)
        panel_label(ax, letter)

    fig.suptitle('GWSA Prediction Validation (2023–2024) — SARIMAX / LSTM / XGBoost',
                 fontsize=11)
    save(fig, 'fig4_prediction_validation')


# ==============================================================================
# FIG 5 — GWSA PROJECTIONS 2025-2030
# ==============================================================================

def fig5_projections():
    """SSP2-4.5 vs SSP5-8.5 projections with 90% PI per zone."""
    hist_path  = OUT_PROC / 'features_master.csv'
    p245_path  = OUT_PROC / 'gwsa_projected_ssp245.csv'
    p585_path  = OUT_PROC / 'gwsa_projected_ssp585.csv'

    if not all(p.exists() for p in [hist_path, p245_path, p585_path]):
        print('  [SKIP] projection files not found')
        return

    df_hist = pd.read_csv(hist_path, parse_dates=['time'])
    df_hist['time'] = pd.to_datetime(df_hist['time'])

    df245 = pd.read_csv(p245_path, parse_dates=['time'])
    df585 = pd.read_csv(p585_path, parse_dates=['time'])
    df245['time'] = pd.to_datetime(df245['time'])
    df585['time'] = pd.to_datetime(df585['time'])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    fig.subplots_adjust(hspace=0.45)

    letters = ['a', 'b', 'c']
    for ax, zone, letter in zip(axes, ZONES, letters):
        dz_hist = df_hist[df_hist['zone'] == zone].sort_values('time')
        if 'gwsa' not in dz_hist.columns:
            continue

        # Historical — clip to end of 2024 to avoid spike artefact
        hist_clip = dz_hist[dz_hist['time'] <= pd.Timestamp('2024-11-30')]
        ax.plot(hist_clip['time'], hist_clip['gwsa'],
                color=PAL['obs'], lw=1.2, label='Historical (observed)', zorder=5)
        ax.axhline(0, color='k', lw=0.4, linestyle=':', alpha=0.5)
        ax.axvline(pd.Timestamp('2025-01-01'), color='gray',
                   lw=1.2, linestyle='--', alpha=0.7, label='Projection start')

        # Projections
        for ssp, df_proj, col, label in [
            ('ssp245', df245, PAL['ssp245'], 'SSP2-4.5'),
            ('ssp585', df585, PAL['ssp585'], 'SSP5-8.5'),
        ]:
            dz_p = df_proj[df_proj['zone'] == zone].sort_values('time') \
                   if 'zone' in df_proj.columns else df_proj.sort_values('time')
            if 'gwsa_mean' not in dz_p.columns:
                continue

            # Connect last historical point to first projection smoothly
            last_hist = hist_clip[hist_clip['zone'] == zone] if 'zone' in hist_clip.columns else hist_clip
            if len(last_hist) > 0 and 'gwsa' in last_hist.columns:
                t_connect = [last_hist['time'].iloc[-1], dz_p['time'].iloc[0]]
                y_connect = [last_hist['gwsa'].iloc[-1], dz_p['gwsa_mean'].iloc[0]]
                ax.plot(t_connect, y_connect, color=col, lw=1.5,
                        linestyle=':', alpha=0.5, zorder=3)
            offset = -0.15 if ssp == 'ssp245' else 0.15
            ax.plot(dz_p['time'], dz_p['gwsa_mean'] + offset,
                    color=col, lw=2.0, label=label, zorder=4)
            if 'gwsa_lo' in dz_p.columns and 'gwsa_hi' in dz_p.columns:
                ax.fill_between(dz_p['time'],
                                dz_p['gwsa_lo'], dz_p['gwsa_hi'],
                                color=col, alpha=0.15,
                                label=f'{label} 90% PI', zorder=3)

        ax.set_ylabel('GWSA (cm EWH)', fontsize=9)
        ax.set_title(ZONE_LABELS[zone], fontsize=10, loc='left', pad=3)
        ax.set_xlabel('Date', fontsize=9)
        ax.legend(fontsize=8, loc='lower left', framealpha=0.7, ncol=2)
        panel_label(ax, letter)

    fig.suptitle('GWSA Projections 2025–2030 under SSP2-4.5 and SSP5-8.5 — Tunisia',
                 fontsize=11)
    save(fig, 'fig5_projections')


# ==============================================================================
# FIG 6 — FEATURE CORRELATION HEATMAP
# ==============================================================================

def fig6_feature_correlations():
    """Feature correlation heatmap from features_master.csv."""
    path = OUT_PROC / 'features_master.csv'
    if not path.exists():
        print('  [SKIP] features_master.csv not found')
        return

    df = pd.read_csv(path)
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ['year', 'month']][:16]

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='Pearson correlation coefficient')

    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(num_cols, fontsize=8)

    # Annotate cells
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color=color)

    ax.set_title('Feature Correlation Matrix — All Zones (2002–2024)',
                 fontsize=11, pad=12)
    fig.tight_layout()
    save(fig, 'fig6_feature_correlations')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(SEP)
    print('09_visualization.py')
    print('Publication-ready figures — 300 DPI')
    print(SEP)

    steps = [
        ('Fig 1 — GWSA time series',         fig1_gwsa_timeseries),
        ('Fig 2 — Trend summary (MMK)',       fig2_trend_summary),
        ('Fig 3 — Water stress',              fig3_water_stress),
        ('Fig 4 — Prediction validation',     fig4_prediction_validation),
        ('Fig 5 — Projections SSP',           fig5_projections),
        ('Fig 6 — Feature correlations',      fig6_feature_correlations),
    ]

    for label, fn in steps:
        print(f'\n[{label}]')
        try:
            fn()
        except Exception as e:
            print(f'  [ERROR] {label}: {e}')

    print('\n' + SEP)
    print('OUTPUTS')
    print(SEP)
    figs = sorted(OUT_FIG.glob('pub_fig*.png'))
    for f in figs:
        size_kb = f.stat().st_size // 1024
        print(f'  {f.name}  ({size_kb} KB)')
    print(f'\n  Total: {len(figs)} figures at 300 DPI')
    print('\n[DONE]')
    print(SEP)


if __name__ == '__main__':
    main()