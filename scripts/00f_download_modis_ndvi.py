# ==============================================================================
# 00f_download_modis_ndvi.py
# Download MODIS MOD13A3 v6.1 NDVI for Tunisia (2002-2024)
# Resolution: 1 km, monthly
# Method: NASA AppEEARS API (area sample)
#
# PREREQUISITE:
#   pip install requests
#   NASA Earthdata account: https://urs.earthdata.nasa.gov/users/new
#   (same account as GLDAS/GRACE)
# ==============================================================================

import os
import sys
import time
import json
import logging
import getpass
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)

import yaml
import requests

with open('config.yaml', 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

OUT_MODIS = Path(CFG['paths']['data']['modis'])
OUT_MODIS.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(CFG['paths']['outputs']['logs'])
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / '00f_download_modis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

APPEEARS_URL = 'https://appeears.earthdatacloud.nasa.gov/api/'

# Tunisia bounding box
LAT_MIN, LAT_MAX =  30.0, 37.5
LON_MIN, LON_MAX =   7.5, 11.6

START_DATE = '01-01-2002'
END_DATE   = '12-31-2024'

TASK_NAME  = 'Tunisia_MOD13A3_NDVI_2002_2024'

# MOD13A3 v6.1 — 1km monthly NDVI
PRODUCT    = 'MOD13A3.061'
LAYER      = '_1_km_monthly_NDVI'


def get_token(username, password):
    """Authenticate with AppEEARS and return bearer token."""
    r = requests.post(
        APPEEARS_URL + 'login',
        auth=(username, password),
        timeout=30
    )
    r.raise_for_status()
    return r.json()['token']


def submit_task(token, task_name):
    """Submit an area sample task for MOD13A3 NDVI over Tunisia."""
    task = {
        "task_type" : "area",
        "task_name" : task_name,
        "params"    : {
            "dates"  : [{"startDate": START_DATE, "endDate": END_DATE}],
            "layers" : [{"product": PRODUCT, "layer": LAYER}],
            "output" : {
                "format"     : {"type": "netcdf4"},
                "projection" : "geographic"
            },
            "geo"    : {
                "type"     : "FeatureCollection",
                "fileName" : "tunisia_bbox",
                "features" : [{
                    "type"       : "Feature",
                    "id"         : "0",
                    "geometry"   : {
                        "type"        : "Polygon",
                        "coordinates" : [[
                            [LON_MIN, LAT_MIN],
                            [LON_MAX, LAT_MIN],
                            [LON_MAX, LAT_MAX],
                            [LON_MIN, LAT_MAX],
                            [LON_MIN, LAT_MIN]
                        ]]
                    },
                    "properties" : {"name": "tunisia"}
                }]
            }
        }
    }
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    r = requests.post(APPEEARS_URL + 'task', json=task, headers=headers, timeout=60)
    if not r.ok:
        log.error(f"  AppEEARS error {r.status_code}: {r.text}")
    r.raise_for_status()
    task_id = r.json()['task_id']
    log.info(f"  Task submitted: {task_id}")
    return task_id


def wait_for_task(token, task_id, poll_interval=60):
    """Poll task status until complete."""
    headers = {'Authorization': f'Bearer {token}'}
    log.info(f"  Polling task {task_id} (this may take 10-60 min for 22 years of 1km data) ...")

    while True:
        r = requests.get(APPEEARS_URL + f'task/{task_id}', headers=headers, timeout=30)
        r.raise_for_status()
        status = r.json()['status']
        pct    = r.json().get('progress', {}).get('summary', '')
        log.info(f"  Status: {status}  {pct}")

        if status == 'done':
            log.info("  Task complete!")
            return True
        elif status in ('error', 'deleted'):
            log.error(f"  Task failed with status: {status}")
            return False

        time.sleep(poll_interval)


def download_results(token, task_id):
    """Download all output files for a completed task."""
    headers = {'Authorization': f'Bearer {token}'}

    # List files
    r = requests.get(APPEEARS_URL + f'bundle/{task_id}', headers=headers, timeout=30)
    r.raise_for_status()
    files = r.json()['files']

    nc_files = [f for f in files if f['file_name'].endswith('.nc')]
    log.info(f"  {len(nc_files)} NetCDF file(s) to download")

    for finfo in nc_files:
        file_id   = finfo['file_id']
        file_name = Path(finfo['file_name']).name
        out_path  = OUT_MODIS / file_name

        if out_path.exists():
            log.info(f"  {file_name} already exists — skipping")
            continue

        log.info(f"  Downloading {file_name} ...")
        dl = requests.get(
            APPEEARS_URL + f'bundle/{task_id}/{file_id}',
            headers=headers, stream=True, timeout=300
        )
        dl.raise_for_status()

        with open(out_path, 'wb') as fout:
            for chunk in dl.iter_content(chunk_size=8192):
                fout.write(chunk)

        size_mb = out_path.stat().st_size / 1024 / 1024
        log.info(f"  [OK] {file_name} ({size_mb:.1f} MB)")


def check_existing_task(token, task_name):
    """Check if a task with this name was already submitted."""
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(APPEEARS_URL + 'task', headers=headers, timeout=30)
    r.raise_for_status()
    tasks = r.json()
    for t in tasks:
        if t.get('task_name') == task_name:
            log.info(f"  Found existing task: {t['task_id']} (status: {t['status']})")
            return t['task_id'], t['status']
    return None, None


def main():
    log.info("=" * 60)
    log.info("00f_download_modis_ndvi.py")
    log.info("=" * 60)
    log.info(f"Product : {PRODUCT} layer {LAYER}")
    log.info(f"Period  : {START_DATE} -> {END_DATE}")
    log.info(f"Area    : lat [{LAT_MIN},{LAT_MAX}], lon [{LON_MIN},{LON_MAX}]")
    log.info(f"Output  : {OUT_MODIS}")

    # Read credentials from .netrc (same file used for GLDAS/GRACE wget)
    # File: C:\Users\RTX\.netrc
    # Expected line: machine urs.earthdata.nasa.gov login argoubi password YOUR_NEW_PASSWORD
    import netrc
    netrc_path = os.path.join(os.path.expanduser('~'), '.netrc')
    try:
        n = netrc.netrc(netrc_path)
        auth = n.authenticators('urs.earthdata.nasa.gov')
        if auth is None:
            raise ValueError("No entry for urs.earthdata.nasa.gov in .netrc")
        username = auth[0]
        password = auth[2]
        log.info(f"  Credentials loaded from .netrc (user: {username})")
    except Exception as e:
        log.error(f"  Could not read .netrc: {e}")
        log.error(f"  Make sure {netrc_path} contains:")
        log.error(f"  machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_NEW_PASSWORD")
        sys.exit(1)

    # Authenticate
    log.info("Authenticating with AppEEARS ...")
    try:
        token = get_token(username, password)
        log.info("  [OK] Authenticated")
    except Exception as e:
        log.error(f"  Authentication failed: {e}")
        sys.exit(1)

    # Check for existing task
    task_id, status = check_existing_task(token, TASK_NAME)

    if task_id is None:
        log.info("Submitting new AppEEARS task ...")
        task_id = submit_task(token, TASK_NAME)
        status  = 'pending'
    else:
        log.info(f"Resuming existing task {task_id} (status: {status})")

    # Wait if not done
    if status != 'done':
        ok = wait_for_task(token, task_id, poll_interval=60)
        if not ok:
            log.error("Task failed. Check AppEEARS dashboard: https://appeears.earthdatacloud.nasa.gov/")
            sys.exit(1)

    # Download results
    log.info("Downloading output files ...")
    download_results(token, task_id)

    # Verify
    nc_files = list(OUT_MODIS.glob('*.nc'))
    log.info(f"\nVerification: {len(nc_files)} NetCDF file(s) in {OUT_MODIS}")

    log.info("\n" + "=" * 60)
    log.info("MODIS NDVI download COMPLETE")
    log.info("Re-run 02_gap_filling.py to use real NDVI data")
    log.info("=" * 60)


if __name__ == '__main__':
    main()