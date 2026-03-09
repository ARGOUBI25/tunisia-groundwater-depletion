"""
check_gldas.py — Vérifie quels fichiers GLDAS manquent pour 2002-2024
"""
import os
from datetime import datetime, date

NOAH_DIR = r"C:\Users\RTX\Desktop\Tunisia_GW\data\gldas\noah"

# Générer tous les mois attendus 2002-2024
expected = []
for year in range(2002, 2025):
    for month in range(1, 13):
        fname = f"GLDAS_NOAH025_M.A{year}{month:02d}.021.nc4.SUB.nc4"
        expected.append((year, month, fname))

# Vérifier lesquels existent et sont valides (>10KB)
present  = []
missing  = []
too_small = []

for year, month, fname in expected:
    fpath = os.path.join(NOAH_DIR, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        if size > 10000:
            present.append(fname)
        else:
            too_small.append((fname, size))
    else:
        missing.append(fname)

print(f"Attendus  : {len(expected)} fichiers (2002-2024)")
print(f"Présents  : {len(present)} fichiers valides")
print(f"Manquants : {len(missing)} fichiers")
print(f"Trop petits: {len(too_small)} fichiers corrompus")

if missing:
    print(f"\nFichiers manquants :")
    for f in missing:
        print(f"  {f}")

if too_small:
    print(f"\nFichiers corrompus (trop petits) :")
    for f, s in too_small:
        print(f"  {f} ({s} bytes)")
