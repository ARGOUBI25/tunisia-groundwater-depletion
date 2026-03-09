# ==============================================================================
# 09b_study_area_map.py
# Tunisia Groundwater Depletion Study
# Figure 1 — Study Area Map (Cartopy / Natural Earth)
# ==============================================================================
# Produces a publication-ready study area map with real Tunisia borders:
#   - Real coastline and borders (Natural Earth 10m)
#   - 3 hydro-climatic zones (North / Central / South)
#   - Major aquifer systems labels
#   - Cities and capital
#   - Precipitation gradient colorbar
#   - Inset: Tunisia in MENA context
#   - Scale bar + North arrow
#
# Requirements: cartopy, matplotlib, numpy
#   pip install cartopy
#
# Output: outputs/figures/pub_fig1_study_area.png (300 DPI)
# ==============================================================================

import os
import sys
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

OUT_FIG = Path('outputs/figures')
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 10,
    'figure.facecolor': 'white',
    'axes.facecolor'  : 'white',
    'savefig.dpi'     : 300,
})

# ── Zone definitions ──────────────────────────────────────────────────────────
ZONE_BOUNDS = {
    'north'  : (34.0, 37.5),
    'central': (32.0, 34.0),
    'south'  : (30.2, 32.0),
}
ZONE_COLORS = {
    'north'  : '#AED6F1',
    'central': '#FAD7A0',
    'south'  : '#A9DFBF',
}
ZONE_ALPHA = 0.55

# Tunisia longitude extent
LON_W, LON_E = 7.4, 11.7
LAT_S, LAT_N = 30.1, 37.6

# ── Aquifer labels ────────────────────────────────────────────────────────────
AQUIFERS = [
    (9.3,  36.6, 'Medjerda\nAlluvial Aq.'),
    (9.9,  35.2, 'Kairouan\nBasin'),
    (8.5,  33.5, 'Complexe\nTerminal (CT)'),
    (9.2,  31.5, 'NWSAS\n(Chotts)'),
    (10.8, 33.6, 'Djeffara\nCoastal Aq.'),
    (8.2,  32.5, 'Continental\nIntercalaire (CI)'),
]

# ── Cities ────────────────────────────────────────────────────────────────────
CITIES = [
    (10.18, 36.82, 'Tunis',    'capital'),
    (10.10, 36.46, 'Nabeul',   'city'),
    (9.56,  35.68, 'Kairouan', 'city'),
    (8.78,  34.74, 'Kasserine','city'),
    (8.78,  33.88, 'Gafsa',    'city'),
    (10.76, 34.74, 'Sfax',     'city'),
    (8.13,  33.50, 'Tozeur',   'city'),
    (10.10, 33.88, 'Gabes',    'city'),
    (10.50, 33.38, 'Medenine', 'city'),
    (9.87,  32.08, 'Tataouine','city'),
]


def add_zone_fills(ax, proj):
    """Add semi-transparent zone fills as horizontal bands."""
    for zone, (lat_s, lat_n) in ZONE_BOUNDS.items():
        # Draw as a filled polygon spanning the full longitude range
        lons = [LON_W - 0.5, LON_E + 0.5, LON_E + 0.5, LON_W - 0.5]
        lats = [lat_s, lat_s, lat_n, lat_n]
        ax.fill(lons, lats,
                transform=ccrs.PlateCarree(),
                color=ZONE_COLORS[zone],
                alpha=ZONE_ALPHA,
                zorder=2)


def add_zone_boundaries(ax):
    """Add dashed lines at zone boundaries."""
    for lat in [32.0, 34.0]:
        ax.plot([LON_W - 0.3, LON_E + 0.3], [lat, lat],
                transform=ccrs.PlateCarree(),
                color='#444444', lw=1.3,
                linestyle='--', zorder=6)


def add_zone_labels(ax):
    """Add zone labels on the left side."""
    labels = {
        'north'  : (7.6, 35.8, 'Northern Zone\n(34°–37.5°N)\nSub-humid'),
        'central': (7.6, 33.0, 'Central Zone\n(32°–34°N)\nSemi-arid'),
        'south'  : (7.6, 31.1, 'Southern Zone\n(30°–32°N)\nArid / Saharan'),
    }
    for zone, (lon, lat, label) in labels.items():
        ax.text(lon, lat, label,
                transform=ccrs.PlateCarree(),
                fontsize=8, ha='left', va='center',
                color='#1a1a1a', zorder=10,
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor='white', alpha=0.82,
                          edgecolor=ZONE_COLORS[zone],
                          linewidth=1.5))


def add_aquifer_labels(ax):
    """Add italic aquifer system labels."""
    for lon, lat, label in AQUIFERS:
        ax.text(lon, lat, label,
                transform=ccrs.PlateCarree(),
                fontsize=7, ha='center', va='center',
                color='#154360', style='italic', zorder=9,
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground='white')])


def add_cities(ax):
    """Plot cities and capital."""
    for lon, lat, name, ctype in CITIES:
        if ctype == 'capital':
            ax.plot(lon, lat, transform=ccrs.PlateCarree(),
                    marker='*', ms=14, color='#c0392b',
                    mec='white', mew=0.8, zorder=11)
        else:
            ax.plot(lon, lat, transform=ccrs.PlateCarree(),
                    marker='o', ms=5, color='#2c3e50',
                    mec='white', mew=0.5, zorder=11)

        offset_x = 0.13 if lon < 10.2 else -0.13
        ha = 'left' if lon < 10.2 else 'right'
        ax.text(lon + offset_x, lat, name,
                transform=ccrs.PlateCarree(),
                fontsize=7.5, ha=ha, va='center', zorder=12,
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground='white')])


def add_scale_bar(ax):
    """Add a 100 km scale bar."""
    # 1° lon ~ 89 km at 34°N → 100 km ~ 1.12°
    x0, y0 = 7.7, 30.75
    x1 = x0 + 1.12
    ax.plot([x0, x1], [y0, y0],
            transform=ccrs.PlateCarree(),
            color='k', lw=3, solid_capstyle='butt', zorder=13)
    ax.plot([x0, x0], [y0 - 0.05, y0 + 0.05],
            transform=ccrs.PlateCarree(), color='k', lw=2, zorder=13)
    ax.plot([x1, x1], [y0 - 0.05, y0 + 0.05],
            transform=ccrs.PlateCarree(), color='k', lw=2, zorder=13)
    ax.text((x0 + x1) / 2, y0 - 0.18, '100 km',
            transform=ccrs.PlateCarree(),
            ha='center', fontsize=8, zorder=13)


def add_north_arrow(ax):
    """Add north arrow."""
    ax.annotate('', xy=(11.35, 34.2), xytext=(11.35, 33.3),
                xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                arrowprops=dict(arrowstyle='->', color='k', lw=2.0))
    ax.text(11.35, 34.4, 'N',
            transform=ccrs.PlateCarree(),
            ha='center', fontsize=11, fontweight='bold', zorder=13)


def draw_main_map(fig):
    """Draw the main Tunisia map with Cartopy."""
    proj = ccrs.PlateCarree()
    ax = fig.add_axes([0.05, 0.08, 0.88, 0.86], projection=proj)
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)

    # ── Natural Earth features ────────────────────────────────────────────────
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),
                   facecolor='#D6EAF8', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),
                   facecolor='#F5F5F0', zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),
                   edgecolor='#555555', linewidth=1.0, zorder=5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                   edgecolor='#333333', linewidth=1.2, zorder=5)
    ax.add_feature(cfeature.LAKES.with_scale('10m'),
                   facecolor='#AED6F1', edgecolor='#5D9EC7',
                   linewidth=0.5, zorder=4)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'),
                   edgecolor='#5D9EC7', linewidth=0.5, zorder=4)

    # Neighbouring country labels
    for lon, lat, name in [(3.0, 28.0, 'ALGERIA'),
                            (15.5, 28.0, 'LIBYA'),
                            (9.5, 38.5, 'Mediterranean Sea')]:
        ax.text(lon, lat, name,
                transform=proj, fontsize=8, ha='center',
                color='#555555', style='italic',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Zone fills (behind land features) ────────────────────────────────────
    add_zone_fills(ax, proj)
    add_zone_boundaries(ax)
    add_zone_labels(ax)
    add_aquifer_labels(ax)
    add_cities(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)

    # ── Gridlines ─────────────────────────────────────────────────────────────
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color='gray', alpha=0.4, linestyle=':')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([8, 9, 10, 11])
    gl.ylocator = mticker.FixedLocator([31, 32, 33, 34, 35, 36, 37])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8.5}
    gl.ylabel_style = {'size': 8.5}

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=ZONE_COLORS['north'],   alpha=0.7,
                       edgecolor='gray', label='Northern Zone (sub-humid, >400 mm/yr)'),
        mpatches.Patch(facecolor=ZONE_COLORS['central'], alpha=0.7,
                       edgecolor='gray', label='Central Zone (semi-arid, 100–400 mm/yr)'),
        mpatches.Patch(facecolor=ZONE_COLORS['south'],   alpha=0.7,
                       edgecolor='gray', label='Southern Zone (arid/Saharan, <100 mm/yr)'),
        Line2D([0],[0], linestyle='--', color='#444444', lw=1.3,
               label='Zone boundary'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='#c0392b',
               markersize=11, label='Capital city (Tunis)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2c3e50',
               markersize=6,  label='Major city'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=8, framealpha=0.9, edgecolor='#aaaaaa',
              title='Legend', title_fontsize=8.5,
              bbox_to_anchor=(0.99, 0.01))

    ax.set_title(
        'Study Area: Tunisia — Three Hydro-climatic Zones and Major Aquifer Systems',
        fontsize=11, pad=10, fontweight='bold')

    return ax


def draw_inset_mena(fig):
    """MENA region inset map."""
    proj = ccrs.PlateCarree()
    ax_in = fig.add_axes([0.06, 0.72, 0.20, 0.18], projection=proj)
    ax_in.set_extent([-10, 60, 10, 45], crs=proj)

    ax_in.add_feature(cfeature.OCEAN.with_scale('50m'),
                      facecolor='#D6EAF8', zorder=0)
    ax_in.add_feature(cfeature.LAND.with_scale('50m'),
                      facecolor='#F0EBD8', zorder=1)
    ax_in.add_feature(cfeature.BORDERS.with_scale('50m'),
                      edgecolor='#888888', linewidth=0.4, zorder=2)
    ax_in.add_feature(cfeature.COASTLINE.with_scale('50m'),
                      edgecolor='#555555', linewidth=0.5, zorder=2)

    # Highlight Tunisia
    ax_in.add_patch(
        mpatches.FancyBboxPatch(
            (7.4, 30.1), 4.3, 7.5,
            boxstyle='square,pad=0',
            transform=proj,
            facecolor='#E74C3C', edgecolor='#C0392B',
            alpha=0.75, lw=1.5, zorder=5
        )
    )
    ax_in.text(9.6, 33.8, 'TUN',
               transform=proj, fontsize=6,
               ha='center', color='white', fontweight='bold', zorder=6)

    ax_in.text(20, 38, 'Mediterranean',
               transform=proj, fontsize=5.5,
               ha='center', color='#2980B9', style='italic')

    ax_in.set_title('MENA Region', fontsize=7, pad=3)
    for spine in ax_in.spines.values():
        spine.set_edgecolor('#666')
        spine.set_linewidth(1.0)

    return ax_in


def main():
    print('Generating Fig 1 — Study Area Map (Cartopy) ...')

    fig = plt.figure(figsize=(11, 12))

    draw_main_map(fig)
    draw_inset_mena(fig)

    out = OUT_FIG / 'pub_fig1_study_area.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] {out}')
    print(f'  Size: {out.stat().st_size // 1024} KB')
    print('[DONE]')


if __name__ == '__main__':
    main()