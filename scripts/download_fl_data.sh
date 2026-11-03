#!/usr/bin/env bash
#SBATCH -J fl_data
#SBATCH -p standard            # any CPU partition with internet access; change if needed
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 24:00:00            # GBIF can take hours to prepare the export
#SBATCH -o /scratch/ajasapar/deepbiosphere/logs/fl_data_%j.log
#
# Downloads the Florida inputs for Deepbiosphere into /scratch/ajasapar/deepbiosphere/data:
#   1. GADM 3.6 US state outlines        -> data/shpfiles/gadm36_USA/
#   2. WorldClim 2.1 BioClim, 30 arc-sec -> data/rasters/bioclim_current/wc_30s_current/
#   3. GBIF plant observations, Florida  -> data/occurrences/
# NAIP imagery is a separate step, once we know which tiles the observations fall in.
#
# Usage (from the login node):
#   mkdir -p /scratch/ajasapar/deepbiosphere/logs
#   GBIF_USER=yourname GBIF_EMAIL=you@example.com sbatch download_fl_data.sh
set -euo pipefail

GBIF_USER="${GBIF_USER:?set GBIF_USER}"
GBIF_EMAIL="${GBIF_EMAIL:?set GBIF_EMAIL}"
AREA="USA.10_1"                       # GADM 3.6 ID for Florida
START="${START:-2015}"                # observation years, inclusive
END="${END:-2025}"

BASE=/scratch/ajasapar/deepbiosphere
DATA="$BASE/data"
REPO="$BASE/Deepbiosphere"
set +u; source "$BASE/activate.sh"; set -u   # conda activate scripts reference unset vars

# ---------- preflight ----------
[ -f ~/.netrc ] || { echo "Missing ~/.netrc with GBIF login (machine api.gbif.org)"; exit 1; }
chmod 600 ~/.netrc
grep -q "api.gbif.org" ~/.netrc || { echo "~/.netrc has no 'machine api.gbif.org' entry"; exit 1; }
curl -sSf -m 20 -o /dev/null https://api.gbif.org/v1/occurrence/search?limit=0 \
  || { echo "This node can't reach GBIF. Try a different partition."; exit 1; }
echo "Free space in scratch:"; df -h /scratch/ajasapar | tail -1

# ---------- patch the --area bug (idempotent) ----------
F="$REPO/src/deepbiosphere/Download_GBIF_Data.py"
if ! grep -q "'--area', type=str, nargs='+'" "$F"; then
  sed -i "s/add_argument('--area', type=str, help=/add_argument('--area', type=str, nargs='+', help=/" "$F"
  echo "Patched --area to accept a list"
fi

# ---------- 1. GADM state outlines ----------
mkdir -p "$DATA/shpfiles/gadm36_USA"
if [ ! -f "$DATA/shpfiles/gadm36_USA/gadm36_USA_1.shp" ]; then
  wget -c -q --show-progress -O "$DATA/shpfiles/gadm36_USA.zip" \
    https://geodata.ucdavis.edu/gadm/gadm3.6/shp/gadm36_USA_shp.zip
  unzip -o -q "$DATA/shpfiles/gadm36_USA.zip" -d "$DATA/shpfiles/gadm36_USA"
  rm "$DATA/shpfiles/gadm36_USA.zip"
fi
python -c "import geopandas as g; s=g.read_file('$DATA/shpfiles/gadm36_USA/gadm36_USA_1.shp'); print('GADM Florida row:', s[s.GID_1=='$AREA'][['NAME_1','HASC_1']].values.tolist())"

# ---------- 2. WorldClim BioClim (19 rasters, ~10 GB zipped) ----------
WC="$DATA/rasters/bioclim_current/wc_30s_current"
mkdir -p "$WC"
if [ "$(ls "$WC"/wc*bio*.tif 2>/dev/null | wc -l)" -ne 19 ]; then
  wget -c -q --show-progress -O "$DATA/rasters/wc2.1_30s_bio.zip" \
    https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_bio.zip
  unzip -o -q "$DATA/rasters/wc2.1_30s_bio.zip" -d "$WC"
  rm "$DATA/rasters/wc2.1_30s_bio.zip"
fi
echo "BioClim rasters: $(ls "$WC"/wc*bio*.tif | wc -l) of 19"

# ---------- 3. GBIF observations (polls every 5 min until ready) ----------
cd "$REPO"   # MEANS path in Utils.py is relative to the repo root
python src/deepbiosphere/Download_GBIF_Data.py \
  --gbif_user "$GBIF_USER" --gbif_email "$GBIF_EMAIL" \
  --organism plant --start_date "$START" --end_date "$END" --area "$AREA"

ls -lh "$DATA/occurrences/"
echo "Done."
