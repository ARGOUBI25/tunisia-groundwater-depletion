"""
sync_github.py
───────────────────────────────────────────────────────────────────────────────
Synchronise automatiquement ton dossier Tunisia_GW → GitHub.
Détecte et uploade TOUT ce qui est dans ton dossier mais pas encore sur GitHub.

Utilisation :
    python sync_github.py              # sync tout + scan nouveaux fichiers
    python sync_github.py --scripts    # sync scripts seulement
    python sync_github.py --figures    # sync figures seulement
    python sync_github.py --results    # sync résultats CSV seulement
    python sync_github.py --paper      # sync LaTeX seulement
    python sync_github.py --check      # voir ce qui changerait sans uploader
    python sync_github.py --no-scan    # sync SYNC_MAP seulement sans scan auto
───────────────────────────────────────────────────────────────────────────────
"""

import sys
import hashlib
import argparse
from pathlib import Path
from github import Github, Auth, GithubException

# ══════════════════════════════════════════════════════════════════════════════
# ← REMPLIR UNE SEULE FOIS
# ══════════════════════════════════════════════════════════════════════════════
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\RTX\Desktop\Tunisia_GW\.env")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_USER  = "ARGOUBI25"
REPO_NAME    = "tunisia-groundwater-depletion"
PROJECT_ROOT = Path(r"C:\Users\RTX\Desktop\Tunisia_GW")
# ══════════════════════════════════════════════════════════════════════════════


# ── Mapping explicite : fichier local → chemin dans le repo ───────────────────
SYNC_MAP = [
    # Scripts
    ("Scripts/00a_setup_project.py",              "scripts/00a_setup_project.py"),
    ("Scripts/00b_download_era5.py",              "scripts/00b_download_era5.py"),
    ("Scripts/00c_download_gldas.py",             "scripts/00c_download_gldas.py"),
    ("Scripts/00d_download_gldas_missing.py",     "scripts/00d_download_gldas_missing.py"),
    ("Scripts/00e_download_vic_missing.py",       "scripts/00e_download_vic_missing.py"),
    ("Scripts/00f_download_modis_ndvi.py",        "scripts/00f_download_modis_ndvi.py"),
    ("Scripts/01_grace_preprocessing.py",         "scripts/01_grace_preprocessing.py"),
    ("Scripts/02_gap_filling.py",                 "scripts/02_gap_filling.py"),
    ("Scripts/03_downscaling.py",                 "scripts/03_downscaling.py"),
    ("Scripts/04_feature_engineering.py",         "scripts/04_feature_engineering.py"),
    ("Scripts/05_ndvi_emulator.py",               "scripts/05_ndvi_emulator.py"),
    ("Scripts/06_trend_analysis.py",              "scripts/06_trend_analysis.py"),
    ("Scripts/07_water_stress_classification.py", "scripts/07_water_stress_classification.py"),
    ("Scripts/08_gwsa_prediction.py",             "scripts/08_gwsa_prediction.py"),
    ("Scripts/09_visualization.py",               "scripts/09_visualization.py"),
    ("Scripts/09b_study_area_map.py",             "scripts/09b_study_area_map.py"),
    ("Scripts/check_gldas.py",                    "scripts/check_gldas.py"),
    # Config
    ("config.yaml",                               "config.yaml"),
    # Résultats
    ("outputs/results/gap_filling_metrics.csv",            "outputs/results/gap_filling_metrics.csv"),
    ("outputs/results/downscaling_metrics.csv",            "outputs/results/downscaling_metrics.csv"),
    ("outputs/results/feature_stats.csv",                  "outputs/results/feature_stats.csv"),
    ("outputs/results/ndvi_emulator_metrics.csv",          "outputs/results/ndvi_emulator_metrics.csv"),
    ("outputs/results/trend_results.csv",                  "outputs/results/trend_results.csv"),
    ("outputs/results/trend_summary.csv",                  "outputs/results/trend_summary.csv"),
    ("outputs/results/stress_classification_metrics.csv",  "outputs/results/stress_classification_metrics.csv"),
    ("outputs/results/prediction_metrics.csv",             "outputs/results/prediction_metrics.csv"),
    # Figures
    ("outputs/figures/pub_fig1_study_area.png",            "outputs/figures/pub_fig1_study_area.png"),
    ("outputs/figures/pub_fig1_gwsa_timeseries.png",       "outputs/figures/pub_fig1_gwsa_timeseries.png"),
    ("outputs/figures/pub_fig2_trend_summary.png",         "outputs/figures/pub_fig2_trend_summary.png"),
    ("outputs/figures/pub_fig3_water_stress.png",          "outputs/figures/pub_fig3_water_stress.png"),
    ("outputs/figures/pub_fig4_prediction_validation.png", "outputs/figures/pub_fig4_prediction_validation.png"),
    ("outputs/figures/pub_fig5_projections.png",           "outputs/figures/pub_fig5_projections.png"),
    ("outputs/figures/pub_fig6_feature_correlations.png",  "outputs/figures/pub_fig6_feature_correlations.png"),
    # Paper LaTeX
    ("paper/abstract_conclusion.tex",      "paper/abstract_conclusion.tex"),
    ("paper/introduction_section.tex",     "paper/introduction_section.tex"),
    ("paper/materials_and_methods_v2.tex", "paper/materials_and_methods_v2.tex"),
    ("paper/results_section.tex",          "paper/results_section.tex"),
    ("paper/discussion_section.tex",       "paper/discussion_section.tex"),
    ("paper/references.bib",               "paper/references.bib"),
]

BINARY_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff",
              ".nc", ".pkl", ".pdf", ".zip", ".h5", ".hdf5"}

# ── Dossiers et fichiers à exclure du scan automatique ────────────────────────
EXCLUDE_DIRS = {
    "data", "__pycache__", ".git", ".idea", ".vscode",
    "venv", "env", ".env", "Lib", "Include",
    "processed", "models", "logs",
}
EXCLUDE_EXT  = {".pyc", ".pyo", ".log", ".tmp", ".cache",
                ".nc", ".pkl", ".h5", ".hdf5"}
EXCLUDE_FILES = {
    "sync_github.py", "setup_github.py", "setup_github_auto.py",
    "desktop.ini", "Thumbs.db", ".DS_Store",
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def sha1_local(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()

def is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXT

def read_file(path: Path):
    if is_binary(path):
        return path.read_bytes()
    return path.read_text(encoding="utf-8", errors="replace")

def log(symbol, msg):
    icons = {"ok":"✓","new":"+","up":"↺","skip":"·","miss":"✗","scan":"🔍","warn":"⚠"}
    print(f"  {icons.get(symbol, symbol)}  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# SCAN AUTOMATIQUE
# ══════════════════════════════════════════════════════════════════════════════

def scan_local_files() -> list:
    """
    Scanne tout PROJECT_ROOT et retourne les fichiers
    qui ne sont PAS encore dans SYNC_MAP.
    """
    already_mapped = {
        str((PROJECT_ROOT / l).resolve())
        for l, r in SYNC_MAP
    }

    new_files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        rel     = path.relative_to(PROJECT_ROOT)
        rel_str = str(rel)
        parts   = rel.parts

        # Exclure dossiers
        if any(p in EXCLUDE_DIRS for p in parts):
            continue
        # Exclure extensions
        if path.suffix.lower() in EXCLUDE_EXT:
            continue
        # Exclure fichiers spéciaux
        if path.name in EXCLUDE_FILES:
            continue
        # Déjà dans SYNC_MAP
        if str(path.resolve()) in already_mapped:
            continue

        repo_path = rel_str.replace("\\", "/")
        new_files.append((rel_str, repo_path))

    return new_files


# ══════════════════════════════════════════════════════════════════════════════
# SYNC
# ══════════════════════════════════════════════════════════════════════════════

def sync_file(repo, local_path: Path, repo_path: str,
              check_only: bool = False) -> str:
    if not local_path.exists():
        log("miss", f"NOT FOUND   {local_path.name}")
        return "missing"

    content   = read_file(local_path)
    local_sha = sha1_local(local_path)

    try:
        remote = repo.get_contents(repo_path)
        if remote.sha == local_sha:
            log("skip", f"unchanged   {repo_path}")
            return "skipped"
        if check_only:
            log("up", f"CHANGED     {repo_path}")
            return "changed"
        repo.update_file(repo_path, f"Update {repo_path}", content, remote.sha)
        log("up", f"updated     {repo_path}")
        return "updated"

    except GithubException:
        if check_only:
            log("new", f"NEW         {repo_path}")
            return "new"
        repo.create_file(repo_path, f"Add {repo_path}", content)
        log("new", f"created     {repo_path}")
        return "created"


def filter_map(args) -> list:
    if args.scripts:
        return [(l, r) for l, r in SYNC_MAP if r.startswith("scripts/")]
    if args.figures:
        return [(l, r) for l, r in SYNC_MAP if r.startswith("outputs/figures/")]
    if args.results:
        return [(l, r) for l, r in SYNC_MAP if r.startswith("outputs/results/")]
    if args.paper:
        return [(l, r) for l, r in SYNC_MAP if r.startswith("paper/")]
    return SYNC_MAP


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sync Tunisia_GW → GitHub")
    parser.add_argument("--scripts",  action="store_true")
    parser.add_argument("--figures",  action="store_true")
    parser.add_argument("--results",  action="store_true")
    parser.add_argument("--paper",    action="store_true")
    parser.add_argument("--check",    action="store_true",
                        help="Voir les changements sans uploader")
    parser.add_argument("--no-scan",  action="store_true",
                        help="Ne pas scanner les nouveaux fichiers")
    args = parser.parse_args()

    if not GITHUB_TOKEN or len(GITHUB_TOKEN) < 10:
        print("⚠  Mets ton token GitHub dans GITHUB_TOKEN en haut du script.")
        sys.exit(1)

    print(f"\n=== Sync Tunisia_GW → GitHub ({GITHUB_USER}/{REPO_NAME}) ===")
    print(f"    Mode : {'CHECK ONLY' if args.check else 'UPLOAD'}\n")

    g    = Github(auth=Auth.Token(GITHUB_TOKEN))
    repo = g.get_user(GITHUB_USER).get_repo(REPO_NAME)

    # Fichiers explicites
    to_sync = filter_map(args)

    # Scan automatique des nouveaux fichiers
    extra_files = []
    mode_filtre = any([args.scripts, args.figures, args.results, args.paper])
    if not args.no_scan and not mode_filtre:
        extra_files = scan_local_files()
        if extra_files:
            print(f"  🔍 {len(extra_files)} nouveaux fichiers détectés"
                  f" (absents de SYNC_MAP) :\n")
            for _, rp in extra_files:
                print(f"       + {rp}")
            print()

    all_files = to_sync + extra_files
    stats = {"created":0, "updated":0, "skipped":0,
             "missing":0, "changed":0, "new":0}

    print("--- Synchronisation en cours ---")
    for local_rel, repo_path in all_files:
        local_path = PROJECT_ROOT / local_rel
        result = sync_file(repo, local_path, repo_path,
                           check_only=args.check)
        stats[result] = stats.get(result, 0) + 1

    created  = stats['created']
    updated  = stats['updated']
    skipped  = stats['skipped']
    missing  = stats['missing']
    total    = len(all_files)

    print(f"""
┌──────────────────────────────────────────────┐
│  Sync terminée — {total:>3} fichiers traités          │
│                                              │
│  +  créés      : {created:>3}                       │
│  ↺  mis à jour : {updated:>3}                       │
│  ·  inchangés  : {skipped:>3}                       │
│  ✗  manquants  : {missing:>3}                       │
└──────────────────────────────────────────────┘
  → https://github.com/{GITHUB_USER}/{REPO_NAME}
""")

    if missing > 0:
        print("  ⚠  Fichiers manquants = pas encore générés.")
        print("     Lance le pipeline puis relance sync_github.py\n")

    if extra_files and not args.check:
        print("  💡 Tip : ajoute ces fichiers dans SYNC_MAP"
              " pour les gérer explicitement.\n")


if __name__ == "__main__":
    main()