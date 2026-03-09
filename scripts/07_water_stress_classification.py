# ==============================================================================
# 07_water_stress_classification.py
# Tunisia Groundwater Depletion Study
# MODULE 6 — Water Stress Classification
# ==============================================================================
# Methodology (Section 2.7):
#   Step 1: K-Means (k=4) exploratory clustering on anomaly features
#           Bootstrap confidence intervals (n=1000)
#   Step 2: Random Forest classifier (primary) trained on K-Means labels
#           LOOCV validation | Features: gwsa_anomaly, precip_anomaly,
#           t2m_anomaly, ndvi_anomaly, spi3, spi12, gwsa_lag1, gwsa_roll12
#   Classes: 0=Low, 1=Moderate, 2=High, 3=Critical stress
#
# Inputs:
#   outputs/processed/features_master.csv
#
# Outputs:
#   outputs/processed/water_stress_labels.csv
#   outputs/processed/water_stress_classified.csv
#   outputs/models/rf_stress_classifier.pkl
#   outputs/results/stress_classification_metrics.csv
#   outputs/figures/07_stress_timeseries.png
#   outputs/figures/07_stress_map_[year].png
#   outputs/figures/07_kmeans_clusters.png
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
        logging.FileHandler(OUT_LOG / '07_water_stress.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
SEP = '=' * 60

K           = CFG['classification']['kmeans']['k']
N_BOOTSTRAP = CFG['classification']['kmeans']['n_bootstrap']
LABELS      = CFG['classification']['kmeans']['labels']
ZONES       = ['north', 'central', 'south']
RANDOM_STATE = CFG['random_state']

# Classification features
CLASS_FEATURES = [
    'gwsa_anomaly', 'precip_anomaly', 't2m_anomaly', 'ndvi_anomaly',
    'spi3', 'spi12', 'gwsa_lag1', 'gwsa_roll12'
]

STRESS_COLORS = {
    0: '#2ecc71',  # low — green
    1: '#f39c12',  # moderate — orange
    2: '#e74c3c',  # high — red
    3: '#8e44ad',  # critical — purple
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_features():
    path = OUT_PROC / 'features_master.csv'
    if not path.exists():
        raise FileNotFoundError("features_master.csv not found. Run 04 first.")
    df = pd.read_csv(path, parse_dates=['time'], index_col='time')
    df.index = df.index.to_period('M').to_timestamp()
    return df


# ==============================================================================
# K-MEANS CLUSTERING
# ==============================================================================

def get_feature_matrix(df):
    """Extract and standardize classification features."""
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in CLASS_FEATURES if c in df.columns]
    X = df[cols + ['zone']].dropna(subset=cols).copy()
    zone_col = X.pop('zone')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler, cols, zone_col


def run_kmeans(X_scaled, k=4, n_init=20):
    """Run K-Means clustering."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=n_init, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    return km, labels


def order_clusters_by_stress(df_feat, labels, gwsa_col='gwsa_anomaly'):
    """
    Re-order cluster labels so that:
    0 = highest GWSA anomaly (least stress)
    3 = lowest GWSA anomaly (most stress)
    """
    cluster_means = {}
    for c in range(K):
        mask = labels == c
        if mask.sum() > 0 and gwsa_col in df_feat.columns:
            cluster_means[c] = df_feat[gwsa_col][mask].mean()
        else:
            cluster_means[c] = 0.0

    # Sort by GWSA mean descending (highest = least stress = 0)
    sorted_clusters = sorted(cluster_means, key=lambda x: -cluster_means[x])
    mapping = {old: new for new, old in enumerate(sorted_clusters)}
    return np.array([mapping[l] for l in labels]), mapping


def bootstrap_cluster_stability(X_scaled, k=4, n_bootstrap=200):
    """
    Bootstrap cluster stability: measure Jaccard similarity across bootstrap samples.
    Returns mean Jaccard similarity (higher = more stable).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    n = len(X_scaled)
    ref_km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    ref_labels = ref_km.fit_predict(X_scaled)

    ari_scores = []
    rng = np.random.RandomState(RANDOM_STATE)
    n_boot = min(n_bootstrap, 200)  # cap for speed

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X_scaled[idx]
        km_boot = KMeans(n_clusters=k, n_init=5, random_state=rng.randint(10000))
        labels_boot = km_boot.fit_predict(X_boot)
        ari = adjusted_rand_score(ref_labels[idx], labels_boot)
        ari_scores.append(ari)

    return np.mean(ari_scores), np.std(ari_scores)


# ==============================================================================
# RANDOM FOREST CLASSIFIER
# ==============================================================================

def train_rf_classifier(X, y, feat_cols):
    """Train RF classifier with cross-validation."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Stratified K-Fold CV (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_macro')
    log.info(f"  CV F1-macro: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Fit on full data
    rf.fit(X, y)
    y_pred = rf.predict(X)

    report = classification_report(y, y_pred,
                                   target_names=[LABELS[i] for i in range(K)],
                                   output_dict=True)
    log.info(f"  Train accuracy: {report['accuracy']:.3f}")
    log.info(f"  F1 per class: " +
             ", ".join(f"{LABELS[i]}={report[LABELS[i]]['f1-score']:.3f}"
                       for i in range(K) if LABELS[i] in report))

    return rf, cv_scores, report


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_stress_timeseries(df_classified, out_path):
    """Figure 07a — Water stress class time series per zone."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, zone in zip(axes, ZONES):
        df_z = df_classified[df_classified['zone'] == zone].sort_index()
        if 'stress_class' not in df_z.columns:
            continue

        # Colored area chart
        for cls in range(K):
            mask = df_z['stress_class'] == cls
            if mask.any():
                ax.fill_between(df_z.index, 0, 1,
                                where=mask.values,
                                color=STRESS_COLORS[cls],
                                alpha=0.6, step='mid')

        # GWSA overlay
        if 'gwsa' in df_z.columns:
            ax2 = ax.twinx()
            ax2.plot(df_z.index, df_z['gwsa'], 'k-', lw=1.0, alpha=0.7)
            ax2.set_ylabel('GWSA (cm EWH)', fontsize=8)
            ax2.axhline(0, color='k', lw=0.5, linestyle=':')

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(zone.capitalize(), fontsize=10, rotation=0, labelpad=40)
        ax.grid(False)

    # Legend
    patches = [mpatches.Patch(color=STRESS_COLORS[i], label=LABELS[i].capitalize())
               for i in range(K)]
    axes[0].legend(handles=patches, loc='upper right', fontsize=9, ncol=4)
    axes[0].set_title('Water Stress Classification — Tunisia Zones (2002-2024)', fontsize=12)
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [OK] {out_path}')


def plot_kmeans_scatter(X_df, labels, feat_cols, out_path):
    """Figure 07b — K-Means cluster scatter (GWSA anomaly vs Precip anomaly)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    xvar = 'gwsa_anomaly' if 'gwsa_anomaly' in feat_cols else feat_cols[0]
    yvar = 'precip_anomaly' if 'precip_anomaly' in feat_cols else feat_cols[1]

    ax = axes[0]
    for cls in range(K):
        mask = labels == cls
        if mask.sum() > 0 and xvar in X_df.columns and yvar in X_df.columns:
            ax.scatter(X_df[xvar][mask], X_df[yvar][mask],
                       color=STRESS_COLORS[cls], alpha=0.5, s=20,
                       label=LABELS[cls].capitalize())
    ax.set_xlabel(xvar.replace('_', ' ').title())
    ax.set_ylabel(yvar.replace('_', ' ').title())
    ax.set_title('K-Means Clusters (k=4)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Class distribution over time
    ax2 = axes[1]
    df_plot = X_df.copy()
    df_plot['stress_class'] = labels
    df_plot['year'] = df_plot.index.year
    counts = df_plot.groupby(['year','stress_class']).size().unstack(fill_value=0)
    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    bottom = np.zeros(len(counts_pct))
    for cls in range(K):
        if cls in counts_pct.columns:
            ax2.bar(counts_pct.index, counts_pct[cls],
                    bottom=bottom, color=STRESS_COLORS[cls],
                    label=LABELS[cls].capitalize(), alpha=0.8)
            bottom += counts_pct[cls].values

    ax2.set_xlabel('Year')
    ax2.set_ylabel('% of zone-months')
    ax2.set_title('Annual Stress Distribution', fontsize=11)
    ax2.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [OK] {out_path}')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(SEP)
    print('07_water_stress_classification.py')
    print(SEP)

    # Load features
    log.info('[STEP 1] Chargement features ...')
    df = load_features()
    log.info(f"  Loaded: {df.shape}")

    # Get feature matrix
    log.info('[STEP 2] Preparation feature matrix ...')
    X_df, X_scaled, scaler, feat_cols, zone_col = get_feature_matrix(df)
    log.info(f"  Features: {feat_cols}")
    log.info(f"  Samples: {len(X_df)} (after dropping NaN)")

    # K-Means clustering
    log.info(f'[STEP 3] K-Means clustering (k={K}) ...')
    km, raw_labels = run_kmeans(X_scaled, k=K)
    ordered_labels, cluster_mapping = order_clusters_by_stress(X_df, raw_labels)
    log.info(f"  Cluster mapping (raw->stress): {cluster_mapping}")

    # Class distribution
    unique, counts = np.unique(ordered_labels, return_counts=True)
    for u, c in zip(unique, counts):
        log.info(f"  Class {u} ({LABELS[u]}): {c} samples ({100*c/len(ordered_labels):.1f}%)")

    # Bootstrap stability
    log.info(f'[STEP 4] Bootstrap stability (n={min(N_BOOTSTRAP, 200)}) ...')
    ari_mean, ari_std = bootstrap_cluster_stability(X_scaled, k=K,
                                                     n_bootstrap=min(N_BOOTSTRAP, 200))
    log.info(f"  ARI: {ari_mean:.3f} +/- {ari_std:.3f} (>0.6 = stable)")

    # Add labels to dataframe
    X_df['stress_class'] = ordered_labels
    X_df['stress_label'] = [LABELS[l] for l in ordered_labels]
    X_df["zone"] = zone_col.values

    # RF classifier
    log.info('[STEP 5] Random Forest classifier ...')
    rf, cv_scores, report = train_rf_classifier(X_scaled, ordered_labels, feat_cols)

    # Save RF model
    model_path = OUT_MOD / 'rf_stress_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': rf, 'scaler': scaler,
            'feat_cols': feat_cols, 'labels': LABELS
        }, f)
    log.info(f'  [OK] {model_path}')

    # Feature importance
    importances = dict(zip(feat_cols, rf.feature_importances_))
    log.info('  Feature importances:')
    for k_feat, v in sorted(importances.items(), key=lambda x: -x[1]):
        log.info(f'    {k_feat}: {v:.3f}')

    # Merge back using positional index on X_df (reset to avoid duplicate timestamp issue)
    X_df_reset = X_df.reset_index()  # 'time' becomes a column
    X_df_reset['stress_class'] = ordered_labels
    X_df_reset['stress_label'] = [LABELS[l] for l in ordered_labels]

    # Rebuild df_classified from X_df_reset
    df_classified = df.copy().reset_index()
    # Match on time + zone
    df_classified = df_classified.merge(
        X_df_reset[['time','zone','stress_class','stress_label']],
        on=['time','zone'], how='left'
    )
    df_classified = df_classified.set_index('time')

    # Fill remaining NaN with RF prediction
    nan_mask = df_classified['stress_class'].isna()
    if nan_mask.sum() > 0:
        X_nan = df_classified.loc[nan_mask, [c for c in feat_cols if c in df_classified.columns]].fillna(0)
        X_nan_scaled = scaler.transform(X_nan)
        pred_labels = rf.predict(X_nan_scaled)
        df_classified.loc[nan_mask, 'stress_class'] = pred_labels
        df_classified.loc[nan_mask, 'stress_label'] = [LABELS[l] for l in pred_labels]

    df_classified['stress_class'] = df_classified['stress_class'].fillna(0).astype(int)

    # Save
    log.info('[STEP 6] Sauvegarde resultats ...')
    df_classified[['zone','gwsa','gwsa_anomaly','stress_class','stress_label']].to_csv(
        OUT_PROC / 'water_stress_classified.csv'
    )
    log.info(f'  [OK] {OUT_PROC}/water_stress_classified.csv')

    X_df.to_csv(OUT_PROC / 'water_stress_labels.csv')
    log.info(f'  [OK] {OUT_PROC}/water_stress_labels.csv')

    # Metrics
    metrics = {
        'k': K,
        'n_samples': len(X_df),
        'ari_mean': ari_mean,
        'ari_std': ari_std,
        'cv_f1_macro_mean': cv_scores.mean(),
        'cv_f1_macro_std': cv_scores.std(),
        'rf_train_accuracy': report['accuracy'],
    }
    for i in range(K):
        label = LABELS[i]
        if label in report:
            metrics[f'f1_{label}'] = report[label]['f1-score']
            metrics[f'n_{label}']  = int(report[label]['support'])

    pd.DataFrame([metrics]).to_csv(OUT_RES / 'stress_classification_metrics.csv', index=False)
    log.info(f'  [OK] {OUT_RES}/stress_classification_metrics.csv')

    # Figures
    log.info('[STEP 7] Generation figures ...')
    plot_stress_timeseries(df_classified, OUT_FIG / '07_stress_timeseries.png')
    plot_kmeans_scatter(X_df, ordered_labels, feat_cols, OUT_FIG / '07_kmeans_clusters.png')

    # Summary
    print('\n' + SEP)
    print('WATER STRESS CLASSIFICATION SUMMARY')
    print(SEP)
    print(f'  K-Means k={K}, Bootstrap ARI={ari_mean:.3f} +/- {ari_std:.3f}')
    print(f'  RF CV F1-macro={cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')
    print(f'  RF Train accuracy={report["accuracy"]:.3f}')
    print()
    print('  Class distribution (all zones):')
    for i in range(K):
        n = (ordered_labels == i).sum()
        print(f'    {i} ({LABELS[i]:10s}): {n:4d} ({100*n/len(ordered_labels):.1f}%)')
    print()
    print('  Stress by zone (most recent 24 months):')
    recent = df_classified[df_classified.index >= df_classified.index.max() -
                           pd.DateOffset(months=24)]
    for zone in ZONES:
        rz = recent[recent['zone'] == zone]
        if len(rz) > 0:
            dom = rz['stress_label'].value_counts().index[0]
            high = (rz['stress_class'] >= 2).mean() * 100
            print(f'    {zone:8s}: dominant={dom}, high/critical={high:.0f}%')

    print()
    print('Outputs:')
    print(f'  {OUT_PROC}/water_stress_classified.csv')
    print(f'  {OUT_PROC}/water_stress_labels.csv')
    print(f'  {OUT_MOD}/rf_stress_classifier.pkl')
    print(f'  {OUT_RES}/stress_classification_metrics.csv')
    print(f'  {OUT_FIG}/07_stress_timeseries.png')
    print(f'  {OUT_FIG}/07_kmeans_clusters.png')
    print()
    print('[DONE] Pret pour 08_gwsa_prediction.py')
    print(SEP)


if __name__ == '__main__':
    main()