#!/usr/bin/env bash
#SBATCH -J label_sum
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/ajasapar/deepbiosphere/logs/label_summary_%j.log
#
# Summary statistics of the Florida labels (runs Build_Data's label steps, no images).
#   sbatch -p preempt --requeue scripts/label_summary.sh
#   THRESHOLD=500 sbatch ...   to preview the paper's 500-observation cutoff
# Output: /scratch/ajasapar/deepbiosphere/data/results/label_summary_fl[_tN]/label_summary_fl.html
set -euo pipefail
BASE=/scratch/ajasapar/deepbiosphere
THRESHOLD="${THRESHOLD:-200}"
OUT="$BASE/data/results/label_summary_fl"
[ "$THRESHOLD" != 200 ] && OUT="${OUT}_t${THRESHOLD}"
set +u; source "$BASE/activate.sh"; set -u   # also cd's into the repo
python "$BASE/scripts/label_summary.py" --threshold "$THRESHOLD" --out "$OUT"
