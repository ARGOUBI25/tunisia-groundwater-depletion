# ==============================================================================
# 06_trend_analysis.py
# Tunisia Groundwater Depletion Study
# MODULE 5 — Trend Detection
# ==============================================================================
# Methodology (Section 2.6):
#   Modified Mann-Kendall (MMK) test with autocorrelation correction (Yue 2002)
#   + Sen's slope estimator for magnitude
#   Applied to: GWSA, Precipitation, Temperature, NDVI
#   Per zone (North, Central, South) and full period (2002-2024)
#   Significance level: alpha=0.05
#
# Inputs:
#   outputs/processed/features_master.csv
#
# Outputs:
#   outputs/results/trend_results.csv
#   outputs/results/trend_summary.csv
#   outputs/figures/06_trend_gwsa.png
#   outputs/figures/06_trend_precip.png
#   outputs/figures/06_trend_ndvi.png
#   outputs/figures/06_trend_summary.png
# ==============================================================================

import os
import sys
import logging
import warnings
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

with open('config.yaml', 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

OUT_RES = Path(CFG['paths']['outputs']['results'])
OUT_FIG = Path(CFG['paths']['outputs']['figures'])
OUT_LOG = Path(CFG['paths']['outputs']['logs'])
OUT_PROC = Path(CFG['paths']['outputs']['processed'])
for p in [OUT_RES, OUT_FIG, OUT_LOG]:
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUT_LOG / '06_trend_analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
SEP = '=' * 60

ALPHA    = CFG['trend']['alpha']
MAX_LAG  = CFG['trend']['max_lag']
ZONES    = ['north', 'central', 'south']
VARIABLES = ['gwsa', 'precip_mm', 't2m_c', 'ndvi']
VAR_LABELS = {
    'gwsa'     : 'GWSA (cm EWH)',
    'precip_mm': 'Precipitation (mm/month)',
    't2m_c'    : 'Temperature (C)',
    'ndvi'     : 'NDVI',
}


# ==============================================================================
# MODIFIED MANN-KENDALL TEST (Yue & Wang 2002)
# ==============================================================================

def autocorr(x, lag):
    """Pearson autocorrelation at given lag."""
    n = len(x)
    if lag >= n:
        return 0.0
    x_mean = np.mean(x)
    num = np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean))
    den = np.sum((x - x_mean)**2)
    return num / den if den > 0 else 0.0


def modified_mann_kendall(series, alpha=0.05, max_lag=12):
    """
    Modified Mann-Kendall test with autocorrelation correction (Yue & Wang 2002).

    Returns dict with:
        tau        : Kendall tau
        p_value    : two-sided p-value (corrected)
        z_score    : standardized test statistic (corrected)
        sen_slope  : Sen's slope (units/month)
        sen_slope_yr: Sen's slope (units/year)
        intercept  : Sen's intercept
        significant: bool (p < alpha)
        trend      : 'increasing' | 'decreasing' | 'no trend'
        n          : sample size
        ns_ratio   : variance inflation factor (n/n*)
    """
    x = np.array(series.dropna(), dtype=float)
    n = len(x)

    if n < 10:
        return {k: np.nan for k in ['tau','p_value','z_score','sen_slope',
                                     'sen_slope_yr','intercept','ns_ratio','n']} | \
               {'significant': False, 'trend': 'insufficient data'}

    # Mann-Kendall S statistic
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(x[j] - x[i])

    # Variance of S (basic)
    var_S = n * (n - 1) * (2 * n + 5) / 18.0

    # Kendall tau
    n_pairs = n * (n - 1) / 2
    tau = S / n_pairs if n_pairs > 0 else 0.0

    # Autocorrelation correction (Yue & Wang 2002)
    # Detrend series first (remove linear trend)
    t = np.arange(n)
    slope_pre, intercept_pre, _, _, _ = stats.linregress(t, x)
    x_detrended = x - (slope_pre * t + intercept_pre)

    # Compute autocorrelation coefficients rho_s(i) for i=1..max_lag
    ns_ratio = 1.0
    for lag in range(1, min(max_lag + 1, n // 4)):
        rho = autocorr(x_detrended, lag)
        ns_ratio += (2 * (n - lag) / n) * rho

    ns_ratio = max(ns_ratio, 1.0)  # cannot be < 1
    var_S_corrected = var_S * ns_ratio

    # Z score
    if S > 0:
        z = (S - 1) / np.sqrt(var_S_corrected)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S_corrected)
    else:
        z = 0.0

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Sen's slope
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (j - i) > 0:
                slopes.append((x[j] - x[i]) / (j - i))
    sen_slope = np.median(slopes) if slopes else 0.0
    sen_slope_yr = sen_slope * 12  # monthly -> annual

    # Sen's intercept (median-based)
    intercept = np.median(x) - sen_slope * np.median(t)

    significant = p_value < alpha
    if significant:
        trend = 'increasing' if S > 0 else 'decreasing'
    else:
        trend = 'no trend'

    return {
        'tau'         : tau,
        'z_score'     : z,
        'p_value'     : p_value,
        'sen_slope'   : sen_slope,
        'sen_slope_yr': sen_slope_yr,
        'intercept'   : intercept,
        'ns_ratio'    : ns_ratio,
        'n'           : n,
        'significant' : significant,
        'trend'       : trend,
    }


# ==============================================================================
# SEN'S TREND LINE
# ==============================================================================

def sen_trend_line(series, sen_slope, intercept):
    """Compute Sen's trend line values for time series."""
    t = np.arange(len(series))
    return pd.Series(intercept + sen_slope * t, index=series.index)


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_trend_variable(var, var_label, zone_series, results_df, out_path):
    """Figure 06x — Time series + trend line for one variable across 3 zones."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'north': '#1f77b4', 'central': '#ff7f0e', 'south': '#2ca02c'}
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    for ax, zone in zip(axes, ZONES):
        s = zone_series[zone]
        ax.plot(s.index, s.values, color=colors[zone], lw=0.9, alpha=0.7,
                label=f'{zone.capitalize()} (observed)')

        # 12-month rolling mean
        roll = s.rolling(12, min_periods=6).mean()
        ax.plot(roll.index, roll.values, color=colors[zone], lw=2.0, alpha=0.95,
                label='12-month rolling mean')

        # Trend line
        row = results_df[(results_df['zone'] == zone) & (results_df['variable'] == var)]
        if len(row) > 0:
            row = row.iloc[0]
            trend_line = sen_trend_line(s.dropna(), row['sen_slope'], row['intercept'])
            ls = '-' if row['significant'] else '--'
            color_t = 'red' if row['trend'] == 'decreasing' else \
                      'green' if row['trend'] == 'increasing' else 'gray'
            ax.plot(trend_line.index, trend_line.values, color=color_t, lw=2.5,
                    linestyle=ls, label=f"Sen slope={row['sen_slope_yr']:.3f}/yr "
                                       f"({'*' if row['significant'] else 'ns'})")

        ax.set_ylabel(var_label, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        if var == 'gwsa':
            ax.axhline(0, color='k', lw=0.5, linestyle=':')

    axes[0].set_title(f'Trend Analysis — {var_label} by Zone (2002-2024)', fontsize=12)
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [OK] {out_path}')


def plot_trend_summary(results_df, out_path):
    """Figure 06 summary — Heatmap of Sen slopes by zone x variable."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sen slope heatmap
    pivot_slope = results_df.pivot(index='zone', columns='variable', values='sen_slope_yr')
    pivot_sig   = results_df.pivot(index='zone', columns='variable', values='significant')

    ax = axes[0]
    vmax = max(abs(pivot_slope.values.max()), abs(pivot_slope.values.min()))
    im = ax.imshow(pivot_slope.values, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(pivot_slope.columns)))
    ax.set_xticklabels([VAR_LABELS.get(c, c) for c in pivot_slope.columns],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot_slope.index)))
    ax.set_yticklabels([z.capitalize() for z in pivot_slope.index], fontsize=10)
    plt.colorbar(im, ax=ax, label='Sen slope (units/year)')
    ax.set_title("Sen's Slope (units/year)", fontsize=11)

    # Add values + significance markers
    for i in range(len(pivot_slope.index)):
        for j in range(len(pivot_slope.columns)):
            val = pivot_slope.values[i, j]
            sig = pivot_sig.values[i, j]
            marker = '*' if sig else ''
            color  = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}{marker}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    # p-value heatmap
    pivot_p = results_df.pivot(index='zone', columns='variable', values='p_value')
    ax2 = axes[1]
    im2 = ax2.imshow(pivot_p.values, cmap='YlOrRd_r', vmin=0, vmax=0.1, aspect='auto')
    ax2.set_xticks(range(len(pivot_p.columns)))
    ax2.set_xticklabels([VAR_LABELS.get(c, c) for c in pivot_p.columns],
                        rotation=30, ha='right', fontsize=9)
    ax2.set_yticks(range(len(pivot_p.index)))
    ax2.set_yticklabels([z.capitalize() for z in pivot_p.index], fontsize=10)
    plt.colorbar(im2, ax=ax2, label='p-value')
    ax2.set_title('p-value (alpha=0.05, dashed line)', fontsize=11)

    for i in range(len(pivot_p.index)):
        for j in range(len(pivot_p.columns)):
            val = pivot_p.values[i, j]
            ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                     fontsize=9, color='black')

    fig.suptitle('Modified Mann-Kendall Trend Analysis — Tunisia Zones (2002-2024)',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [OK] {out_path}')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(SEP)
    print('06_trend_analysis.py')
    print(SEP)

    # Load features
    log.info('[STEP 1] Chargement features_master.csv ...')
    master_path = OUT_PROC / 'features_master.csv'
    if not master_path.exists():
        raise FileNotFoundError("features_master.csv not found. Run 04 first.")

    df = pd.read_csv(master_path, parse_dates=['time'], index_col='time')
    df.index = df.index.to_period('M').to_timestamp()
    log.info(f"  Loaded: {df.shape}")

    # Run MMK test for each variable x zone
    log.info('[STEP 2] Modified Mann-Kendall tests ...')
    results = []
    zone_series_by_var = {v: {} for v in VARIABLES}

    for zone in ZONES:
        df_z = df[df['zone'] == zone].sort_index()
        log.info(f'\n  -- Zone: {zone.upper()} ({len(df_z)} months) --')

        for var in VARIABLES:
            if var not in df_z.columns:
                log.warning(f'  Variable {var} not found in features')
                continue

            s = df_z[var].dropna()
            if len(s) < 10:
                log.warning(f'  {var}: insufficient data ({len(s)})')
                continue

            zone_series_by_var[var][zone] = s

            mmk = modified_mann_kendall(s, alpha=ALPHA, max_lag=MAX_LAG)
            row = {
                'zone'        : zone,
                'variable'    : var,
                'n'           : mmk['n'],
                'tau'         : mmk['tau'],
                'z_score'     : mmk['z_score'],
                'p_value'     : mmk['p_value'],
                'sen_slope'   : mmk['sen_slope'],
                'sen_slope_yr': mmk['sen_slope_yr'],
                'intercept'   : mmk['intercept'],
                'ns_ratio'    : mmk['ns_ratio'],
                'significant' : mmk['significant'],
                'trend'       : mmk['trend'],
            }
            results.append(row)

            sig_str = '*** SIGNIFICANT ***' if mmk['significant'] else 'not significant'
            log.info(f"  {var:12s}: tau={mmk['tau']:+.3f}, "
                     f"p={mmk['p_value']:.4f}, "
                     f"slope={mmk['sen_slope_yr']:+.4f}/yr, "
                     f"{sig_str}")

    results_df = pd.DataFrame(results)

    # Save results
    log.info('\n[STEP 3] Sauvegarde resultats ...')
    results_df.to_csv(OUT_RES / 'trend_results.csv', index=False)
    log.info(f'  [OK] {OUT_RES}/trend_results.csv')

    # Summary table
    summary = results_df[results_df['significant'] == True][
        ['zone','variable','sen_slope_yr','p_value','trend']
    ].sort_values(['variable','zone'])
    summary.to_csv(OUT_RES / 'trend_summary.csv', index=False)
    log.info(f'  [OK] {OUT_RES}/trend_summary.csv')

    # Plots
    log.info('[STEP 4] Generation figures ...')
    for var in VARIABLES:
        if not zone_series_by_var[var]:
            continue
        label = VAR_LABELS.get(var, var)
        plot_trend_variable(
            var, label,
            zone_series_by_var[var],
            results_df,
            OUT_FIG / f'06_trend_{var}.png'
        )

    plot_trend_summary(results_df, OUT_FIG / '06_trend_summary.png')

    # Print summary
    print('\n' + SEP)
    print('TREND ANALYSIS RESULTS')
    print(SEP)
    print(results_df[['zone','variable','tau','p_value','sen_slope_yr','trend']
          ].to_string(index=False, float_format='{:.4f}'.format))

    print('\n' + SEP)
    print('SIGNIFICANT TRENDS (p < 0.05)')
    print(SEP)
    sig = results_df[results_df['significant'] == True]
    if len(sig) > 0:
        print(sig[['zone','variable','sen_slope_yr','p_value','trend']
              ].to_string(index=False, float_format='{:.4f}'.format))
    else:
        print('  No significant trends detected')

    print('\n' + SEP)
    print('RESUME 06_trend_analysis.py')
    print(SEP)
    print(f'  Variables   : {VARIABLES}')
    print(f'  Zones       : {ZONES}')
    print(f'  Method      : Modified Mann-Kendall (Yue & Wang 2002)')
    print(f'  Alpha       : {ALPHA}')
    print(f'  Max lag     : {MAX_LAG}')
    print(f'  Significant : {len(sig)}/{len(results_df)} tests')
    print()
    print('Outputs:')
    print(f'  {OUT_RES}/trend_results.csv')
    print(f'  {OUT_RES}/trend_summary.csv')
    for var in VARIABLES:
        print(f'  {OUT_FIG}/06_trend_{var}.png')
    print(f'  {OUT_FIG}/06_trend_summary.png')
    print()
    print('[DONE] Pret pour 07_water_stress_classification.py')
    print(SEP)


if __name__ == '__main__':
    main()
