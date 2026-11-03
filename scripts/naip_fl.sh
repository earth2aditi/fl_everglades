#!/usr/bin/env bash
#SBATCH -J naip_fl
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -o /scratch/ajasapar/deepbiosphere/logs/naip_fl_%j.log
#
# Prepares Florida 2021 NAIP imagery for Deepbiosphere.
#
#   Step 1 (plan, ~15-30 min):   sbatch scripts/naip_fl.sh
#     - patches the repo for Florida (UTM 17N, Florida split box)
#     - downloads EPA Level III ecoregions for Florida
#     - finds the NAIP tiles that contain observations and reports the download size
#
#   Step 2 (download, hours):    MODE=download sbatch scripts/naip_fl.sh
#     optional: TOP=800 to fetch only the 800 busiest tiles; MAX_GB=300 to raise the size guard
set -euo pipefail

MODE="${MODE:-plan}"
BASE=/scratch/ajasapar/deepbiosphere
DATA="$BASE/data"
REPO="$BASE/Deepbiosphere"
SCRIPTS="$BASE/scripts"

set +u; source "$BASE/activate.sh"; set -u   # conda activate scripts reference unset vars
python -c "import pystac_client, planetary_computer" 2>/dev/null \
  || pip install -q pystac-client planetary-computer

if [ "$MODE" = "plan" ]; then
  echo "== 1. Patch repo for Florida"
  python "$SCRIPTS/patch_florida.py" "$REPO"

  echo "== 2. EPA Level III ecoregions (Florida)"
  ECO="$DATA/shpfiles/ecoregions/fl"
  if [ ! -f "$ECO/fl_eco_l3.shp" ]; then
    mkdir -p "$ECO"
    wget -q -O "$ECO/fl_eco_l3.zip" \
      https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/fl/fl_eco_l3.zip
    unzip -o -q "$ECO/fl_eco_l3.zip" -d "$ECO"
    # flatten if the zip has a subfolder
    find "$ECO" -mindepth 2 -name 'fl_eco_l3.*' -exec mv {} "$ECO/" \;
    rm "$ECO/fl_eco_l3.zip"
  fi
  python -c "import geopandas as g; e=g.read_file('$ECO/fl_eco_l3.shp'); print(len(e), 'ecoregion polygons;', sorted(e.US_L3NAME.unique()))"

  echo "== 3. NAIP plan"
  python "$SCRIPTS/naip_fl.py" plan --workers 16
else
  echo "== NAIP download"
  python "$SCRIPTS/naip_fl.py" download --workers 8 --top "${TOP:-0}" --max_gb "${MAX_GB:-120}"
  echo "Tiles on disk: $(find "$DATA/scratch/fl_060cm_2021" -name '*.tif' | wc -l)"
  du -sh "$DATA/scratch/fl_060cm_2021"
fi
echo "Done."
