"""
00e_download_vic_missing.py
Télécharge uniquement les fichiers GLDAS VIC manquants (2002-2024)
"""

import os
import requests
import time

VIC_DIR  = r"C:\Users\RTX\Desktop\Tunisia_GW\data\gldas\vic"
USERNAME = "argoubi"
PASSWORD = "LXs=%7=b,L!vg.h"

# ── Identifier les fichiers manquants ─────────────────────────────────────────
missing = []
for year in range(2002, 2025):
    for month in range(1, 13):
        fname = f"GLDAS_VIC10_M.A{year}{month:02d}.021.nc4.SUB.nc4"
        fpath = os.path.join(VIC_DIR, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 5000:
            missing.append((year, month, fname))

print(f"Fichiers manquants : {len(missing)}")
if len(missing) == 0:
    print("Tous les fichiers VIC sont présents !")
    exit()

# ── URLs VIC ──────────────────────────────────────────────────────────────────
BASE_URL = (
    "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi"
    "?FILENAME=%2Fdata%2FGLDAS%2FGLDAS_VIC10_M.2.1%2F{year}%2F"
    "GLDAS_VIC10_M.A{year}{month:02d}.021.nc4"
    "&VARIABLES=SoilMoi0_30cm_inst"
    "&DATASET_VERSION=2.1&SERVICE=L34RS_LDAS&FORMAT=bmM0Lw&VERSION=1.02"
    "&BBOX=29.601%2C6.965%2C37.95%2C12.458&SHORTNAME=GLDAS_VIC10_M"
    "&LABEL=GLDAS_VIC10_M.A{year}{month:02d}.021.nc4.SUB.nc4"
)

# ── Session ───────────────────────────────────────────────────────────────────
session = requests.Session()
session.auth = (USERNAME, PASSWORD)
session.headers.update({"User-Agent": "Mozilla/5.0"})

success = 0
failed  = []

for i, (year, month, fname) in enumerate(missing, 1):
    url      = BASE_URL.format(year=year, month=month)
    out_path = os.path.join(VIC_DIR, fname)

    print(f"[{i:02d}/{len(missing)}] {fname} ...", end=" ", flush=True)

    try:
        time.sleep(2)
        resp = session.get(url, allow_redirects=True, timeout=120)

        if len(resp.content) < 5000:
            time.sleep(30)
            resp = session.get(url, allow_redirects=True, timeout=120)

        if len(resp.content) < 5000:
            print(f"ECHEC ({len(resp.content)} bytes)")
            failed.append(fname)
            continue

        with open(out_path, "wb") as f:
            f.write(resp.content)

        print(f"OK ({len(resp.content)//1024} KB)")
        success += 1

    except Exception as e:
        print(f"ERREUR: {e}")
        failed.append(fname)
        time.sleep(10)

print("\n" + "=" * 50)
print(f"Téléchargés : {success}/{len(missing)}")
if failed:
    print(f"Échecs : {len(failed)}")
    for f in failed:
        print(f"  {f}")
else:
    print("Tous les fichiers VIC manquants téléchargés !")
print("=" * 50)
