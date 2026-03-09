"""
00c_download_gldas.py
Téléchargement automatique des 313 fichiers GLDAS NOAH
depuis la liste de liens GES DISC
"""

import os
import requests
import netrc
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
LINKS_FILE = r"C:\Users\RTX\Desktop\Tunisia_GW\data\gldas\noah\links.txt"
OUT_DIR    = r"C:\Users\RTX\Desktop\Tunisia_GW\data\gldas\noah"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Authentification NASA Earthdata ───────────────────────────────────────────
netrc_path = os.path.join(os.path.expanduser("~"), ".netrc")
auth = netrc.netrc(netrc_path).authenticators("urs.earthdata.nasa.gov")
USERNAME = auth[0]
PASSWORD = auth[2]

print(f"Connecté en tant que : {USERNAME}")

# ── Lecture des liens ─────────────────────────────────────────────────────────
with open(LINKS_FILE, "r") as f:
    links = [line.strip() for line in f.readlines()
             if line.strip().startswith("http") and "GLDAS" in line]

print(f"Nombre de fichiers à télécharger : {len(links)}")

# ── Téléchargement ────────────────────────────────────────────────────────────
session = requests.Session()
session.auth = (USERNAME, PASSWORD)

success = 0
failed  = []

for i, url in enumerate(links, 1):
    # Extraire le nom du fichier depuis le paramètre LABEL dans l'URL
    filename = None
    for part in url.split("&"):
        if part.startswith("LABEL="):
            filename = part.replace("LABEL=", "")
            break
    if filename is None:
        filename = f"gldas_{i:04d}.nc4"

    out_path = os.path.join(OUT_DIR, filename)

    # Sauter si déjà téléchargé
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"[SKIP {i:03d}/{len(links)}] {filename}")
        success += 1
        continue

    print(f"[DOWN {i:03d}/{len(links)}] {filename} ...", end=" ")

    try:
        response = session.get(url, timeout=120)
        response.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(response.content)

        size_kb = os.path.getsize(out_path) / 1024
        print(f"OK ({size_kb:.0f} KB)")
        success += 1

    except Exception as e:
        print(f"ERREUR: {e}")
        failed.append((i, url, str(e)))

    # Pause pour ne pas surcharger le serveur
    time.sleep(0.5)

# ── Résumé ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"Téléchargés avec succès : {success}/{len(links)}")
if failed:
    print(f"Échecs ({len(failed)}) :")
    for idx, url, err in failed:
        print(f"  [{idx}] {err}")
    # Sauvegarder les liens échoués pour retry
    retry_file = os.path.join(OUT_DIR, "failed_links.txt")
    with open(retry_file, "w") as f:
        for _, url, _ in failed:
            f.write(url + "\n")
    print(f"  Liens échoués sauvegardés dans: {retry_file}")
else:
    print("Tous les fichiers téléchargés avec succès !")
print("=" * 50)
