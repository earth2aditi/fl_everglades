#!/usr/bin/env bash
#SBATCH -J fl_figs
#SBATCH -p standard
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o /scratch/ajasapar/deepbiosphere/logs/figures_fl_%j.log
#
# Descriptive figures for the Florida data (GBIF, BioClim, ecoregions, NAIP). No Build_Data needed.
#   sbatch -p preempt --requeue scripts/figures_fl.sh
#   sbatch scripts/figures_fl.sh --only years,season        # just some figures
# Output: /scratch/ajasapar/deepbiosphere/data/results/figures_fl/figures_fl.html (+ png/, tables/)
set -euo pipefail
BASE=/scratch/ajasapar/deepbiosphere
set +u; source "$BASE/activate.sh"; set -u
python "$BASE/scripts/figures_fl.py" "$@"
