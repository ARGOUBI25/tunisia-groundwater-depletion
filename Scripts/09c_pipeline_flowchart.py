# ==============================================================================
# 09c_pipeline_flowchart.py
# Tunisia Groundwater Depletion Study
# Figure 2 — Methodology Pipeline Flowchart
# ==============================================================================
# Produces a publication-ready flowchart of the 9-module pipeline:
#   - Data inputs (GRACE, ERA5, MODIS, GLDAS, CMIP6, DGRE)
#   - 9 processing modules with outputs
#   - Colour-coded by module type
#   - Arrows showing data flow
#
# Requirements: matplotlib, numpy
# Output: outputs/figures/pub_fig2_pipeline_flowchart.png (300 DPI)
# ==============================================================================

import os
import sys
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

OUT_FIG = Path('outputs/figures')
OUT_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 9,
    'figure.facecolor': 'white',
    'axes.facecolor' : 'white',
})

# ── Colour scheme ─────────────────────────────────────────────────────────────
C = {
    'data'    : '#D6EAF8',   # light blue   — input data
    'preproc' : '#D5F5E3',   # light green  — preprocessing
    'ml'      : '#FEF9E7',   # light yellow — ML / modelling
    'output'  : '#FDEDEC',   # light pink   — outputs / results
    'arrow'   : '#555555',
    'border_data'    : '#2980B9',
    'border_preproc' : '#27AE60',
    'border_ml'      : '#F39C12',
    'border_output'  : '#E74C3C',
}

def box(ax, x, y, w, h, label, sublabel='', color='#FFFFFF',
        border='#333333', fontsize=8.5, bold=False):
    """Draw a rounded rectangle with label."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.02',
                          facecolor=color, edgecolor=border,
                          linewidth=1.3, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=4, wrap=True)
    if sublabel:
        ax.text(x, y - 0.028, sublabel,
                ha='center', va='center', fontsize=6.8,
                color='#555555', zorder=4, style='italic')

def arrow(ax, x1, y1, x2, y2, color='#555555', style='->', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style,
                                color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'))

def main():
    print('Generating Fig 2 — Pipeline Flowchart ...')

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ── TITLE ─────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97,
            'Methodological Framework — Satellite-Based Groundwater Analysis Pipeline',
            ha='center', va='top', fontsize=11, fontweight='bold')

    # ── LEGEND ────────────────────────────────────────────────────────────────
    legend_items = [
        (C['data'],    C['border_data'],    'Input Data'),
        (C['preproc'], C['border_preproc'], 'Pre-processing & Feature Engineering'),
        (C['ml'],      C['border_ml'],      'ML / Statistical Modelling'),
        (C['output'],  C['border_output'],  'Outputs & Results'),
    ]
    lx = 0.01
    for fc, ec, label in legend_items:
        rect = FancyBboxPatch((lx, 0.005), 0.025, 0.025,
                              boxstyle='round,pad=0.01',
                              facecolor=fc, edgecolor=ec, lw=1.0, zorder=3)
        ax.add_patch(rect)
        ax.text(lx + 0.030, 0.018, label, va='center', fontsize=7.5)
        lx += 0.22

    # ==========================================================================
    # ROW 1 — INPUT DATA SOURCES
    # ==========================================================================
    y_data = 0.88
    bw, bh = 0.13, 0.065

    data_sources = [
        (0.09,  'GRACE/GRACE-FO\nJPL RL06.3Mv04',   '252 monthly fields'),
        (0.25,  'ERA5 Reanalysis\nP & T2m',           '2002–2024'),
        (0.41,  'MODIS MOD13A3\nNDVI 1km',            '2002–2024'),
        (0.57,  'GLDAS NOAH\n& VIC',                  'SM, SWE, SW'),
        (0.73,  'CMIP6\nSSP2-4.5 / SSP5-8.5',         '6 GCMs'),
        (0.89,  'DGRE\nPiezometric data',              'In-situ wells'),
    ]
    for x, label, sub in data_sources:
        box(ax, x, y_data, bw, bh, label, sub,
            color=C['data'], border=C['border_data'], fontsize=7.5)

    # ── Section label ─────────────────────────────────────────────────────────
    ax.text(0.005, y_data, 'INPUT\nDATA', ha='left', va='center',
            fontsize=7, color=C['border_data'], fontweight='bold',
            rotation=90)

    # ==========================================================================
    # ROW 2 — PREPROCESSING MODULES (01-04)
    # ==========================================================================
    y_pp = 0.72
    bw2, bh2 = 0.155, 0.068

    preproc_modules = [
        (0.10, '01 GRACE\nPre-processing',
               'Ensemble GWSA\ngwsa_ensemble.nc'),
        (0.29, '02 Gap Filling\nLSTM-BCNN',
               'Continuous 2002–2024\ngwsa_gap_filled.nc'),
        (0.50, '03 Spatial\nDownscaling RF',
               '1 km GWSA grid\ngwsa_1km.nc'),
        (0.71, '04 Feature\nEngineering',
               '37 features / 750 rows\nfeatures_master.csv'),
        (0.90, '05 NDVI\nEmulator RF',
               'Future NDVI (SSP)\nndvi_emulated_*.csv'),
    ]
    for x, label, sub in preproc_modules:
        box(ax, x, y_pp, bw2, bh2, label, sub,
            color=C['preproc'], border=C['border_preproc'], fontsize=7.5)

    ax.text(0.005, y_pp, 'PRE-\nPROC', ha='left', va='center',
            fontsize=7, color=C['border_preproc'], fontweight='bold',
            rotation=90)

    # ── Arrows: data → preproc ────────────────────────────────────────────────
    # GRACE → 01
    arrow(ax, 0.09, y_data - bh/2, 0.10, y_pp + bh2/2)
    # ERA5 → 02, 04
    arrow(ax, 0.25, y_data - bh/2, 0.29, y_pp + bh2/2)
    # MODIS → 03, 04
    arrow(ax, 0.41, y_data - bh/2, 0.50, y_pp + bh2/2)
    # GLDAS → 01
    arrow(ax, 0.57, y_data - bh/2, 0.10, y_pp + bh2/2, color='#aaaaaa')
    # CMIP6 → 05
    arrow(ax, 0.73, y_data - bh/2, 0.90, y_pp + bh2/2)
    # DGRE → 04
    arrow(ax, 0.89, y_data - bh/2, 0.71, y_pp + bh2/2, color='#aaaaaa')

    # ── Arrows: 01 → 02 → 03 → 04 ────────────────────────────────────────────
    arrow(ax, 0.10 + bw2/2, y_pp, 0.29 - bw2/2, y_pp, color='#27AE60', lw=1.5)
    arrow(ax, 0.29 + bw2/2, y_pp, 0.50 - bw2/2, y_pp, color='#27AE60', lw=1.5)
    arrow(ax, 0.50 + bw2/2, y_pp, 0.71 - bw2/2, y_pp, color='#27AE60', lw=1.5)

    # ==========================================================================
    # ROW 3 — MODELLING MODULES (06-08)
    # ==========================================================================
    y_ml = 0.52
    bw3, bh3 = 0.19, 0.072

    ml_modules = [
        (0.18, '06 Trend Analysis\nModified Mann-Kendall',
               'Sen slopes · p-values\ntrend_results.csv'),
        (0.50, '07 Water Stress\nClassification K-Means / RF',
               'Stress labels · F1=0.957\nwater_stress_classified.csv'),
        (0.82, '08 GWSA Prediction\nSARIMAX + LSTM + XGBoost',
               'Validation + SSP projections\ngwsa_projected_*.csv'),
    ]
    for x, label, sub in ml_modules:
        box(ax, x, y_ml, bw3, bh3, label, sub,
            color=C['ml'], border=C['border_ml'], fontsize=7.5)

    ax.text(0.005, y_ml, 'MODEL-\nLING', ha='left', va='center',
            fontsize=7, color=C['border_ml'], fontweight='bold',
            rotation=90)

    # ── Arrows: preproc → modelling ───────────────────────────────────────────
    arrow(ax, 0.29, y_pp - bh2/2, 0.18, y_ml + bh3/2)   # 02 → 06
    arrow(ax, 0.71, y_pp - bh2/2, 0.50, y_ml + bh3/2)   # 04 → 07
    arrow(ax, 0.71, y_pp - bh2/2, 0.82, y_ml + bh3/2)   # 04 → 08
    arrow(ax, 0.90, y_pp - bh2/2, 0.82, y_ml + bh3/2)   # 05 → 08

    # ==========================================================================
    # ROW 4 — KEY RESULTS
    # ==========================================================================
    y_out = 0.30
    bw4, bh4 = 0.19, 0.085

    results = [
        (0.18, 'Trend Results',
               u'GWSA: \u22120.31 to \u22120.53 cm EWH/yr\nAll zones p < 0.001 ***'),
        (0.50, 'Stress Classification',
               'North: 92% High/Critical\n(2023\u20132024)'),
        (0.82, 'Projections 2025\u20132030',
               'Continued depletion\nSSP2-4.5 & SSP5-8.5'),
    ]
    for x, label, sub in results:
        box(ax, x, y_out, bw4, bh4, label, sub,
            color=C['output'], border=C['border_output'],
            fontsize=8, bold=True)

    ax.text(0.005, y_out, 'KEY\nRESULTS', ha='left', va='center',
            fontsize=7, color=C['border_output'], fontweight='bold',
            rotation=90)

    # ── Arrows: modelling → results ───────────────────────────────────────────
    for x in [0.18, 0.50, 0.82]:
        arrow(ax, x, y_ml - bh3/2, x, y_out + bh4/2,
              color='#E74C3C', lw=1.5)

    # ==========================================================================
    # ROW 5 — FINAL OUTPUT (Fig 09)
    # ==========================================================================
    y_fig = 0.12
    box(ax, 0.50, y_fig, 0.55, 0.060,
        '09  Publication-ready Figures (300 DPI)',
        '6 figures  \u2022  Uniform style  \u2022  Open data via GitHub + Zenodo DOI',
        color='#F4F6F7', border='#7F8C8D', fontsize=8.5, bold=True)

    arrow(ax, 0.18, y_out - bh4/2, 0.35, y_fig + 0.030,
          color='#7F8C8D', lw=1.2)
    arrow(ax, 0.50, y_out - bh4/2, 0.50, y_fig + 0.030,
          color='#7F8C8D', lw=1.2)
    arrow(ax, 0.82, y_out - bh4/2, 0.65, y_fig + 0.030,
          color='#7F8C8D', lw=1.2)

    # ── Module numbers in corner ───────────────────────────────────────────────
    # Already embedded in box labels above

    # ── Validation badge ──────────────────────────────────────────────────────
    badge_x, badge_y = 0.50, 0.425
    circ = plt.Circle((badge_x, badge_y), 0.028,
                       color='#F39C12', zorder=6, alpha=0.15)
    ax.add_patch(circ)
    ax.text(badge_x, badge_y,
            'LOYO\nCV',
            ha='center', va='center', fontsize=6.5,
            color='#7d5a00', zorder=7, fontweight='bold')

    out = OUT_FIG / 'pub_fig2_pipeline_flowchart.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] {out}')
    print(f'  Size: {out.stat().st_size // 1024} KB')
    print('[DONE]')


if __name__ == '__main__':
    main()
