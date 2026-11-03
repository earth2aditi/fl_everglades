# Deepbiosphere: Florida

Adapts [Deepbiosphere](patches/UPSTREAM.txt) (species distribution modeling from NAIP aerial imagery + citizen-science occurrences) from California to Florida. The pipeline runs on BlueHive 3 (University of Rochester CIRC).

## Layout

| Path | What |
|---|---|
| `activate.sh` | `source` it to load the `deepbio` env (Miniforge, Python 3.10, torch 2.3.1) |
| `scripts/` | Data download, NAIP tiling, repo patching, label summary, figures (Slurm wrappers + Python) |
| `patches/florida.patch` | Florida changes to upstream Deepbiosphere (see below) |
| `patches/UPSTREAM.txt` | Upstream remote and commit the patch applies to |
| `data/occurrences/fl_species_min200.json` | Species list for `Build_Data.py --species_file` (691 species, ≥200 records) |
| `data/results/` | Label summary report and figures (small outputs only) |

Everything else under `data/` (GBIF CSV, NAIP tiles, rasters, shapefiles) is git-ignored and recreated by the scripts.

## Upstream patches

`Deepbiosphere/` is a clone of upstream, installed editable, and is not committed here. The Florida changes:

- `Download_GBIF_Data.py`: `--area` takes `nargs='+'` (a single string was being split into characters)
- `NAIP_Utils.py`: `NAIP_CRS_1/2` set to EPSG:26917 (UTM 17N)
- `Build_Data.py`: spatial split box set to Florida (lon −88..−79.5, lat 24..31.1)

To apply them to a fresh clone: `patch -p1 -d Deepbiosphere < patches/florida.patch` (or run `scripts/patch_florida.py`).

## Data sources

- **Occurrences:** GBIF, Plantae, Florida (GADM USA.10_1), 2015–2025, human observations, coordinate uncertainty ≤120 m, no geospatial issues. 739,764 records, 4,325 species, 96% iNaturalist. **GBIF download ID 0012182-261023074918325** (cite this).
- **Imagery:** NAIP program year 2021, 60 cm, Florida (some tiles were flown in January 2022).
- **Climate:** WorldClim 2.1 BioClim, 30 arc-second.
- **Boundaries:** GADM 3.6; EPA Level III ecoregions.

## Pipeline (in order)

1. `scripts/setup_deepbiosphere.sh`: env + editable install; then `scripts/patch_florida.py`
2. `scripts/download_fl_data.sh`: GBIF, GADM, WorldClim, ecoregions
3. `scripts/naip_fl.sh` / `naip_fl.py`: plan and download NAIP tiles, build the tile index
4. `scripts/label_summary.sh`: filtering funnel + checks → `data/results/label_summary_fl/`
5. `scripts/figures_fl.sh`: figures → `data/results/figures_fl/`
6. `Build_Data.py --state fl --year 2021 --species_file data/occurrences/fl_species_min200.json --sep '\t' --idCol gbifID --add_images --parallel N`

Use `sbatch -p preempt --requeue` for restartable jobs.

## Status

- Done: data downloads, 1,822 NAIP tiles, label summary (263,490 obs / 687 species inside current tile coverage), figures.
- In progress: NAIP top-up. 42% of records fell outside the tile set because the 2021 program-year search dropped tiles flown in 2022; `naip_fl.py` now searches 2021–2022.
- Next: rerun label summary and figures, pick the species cutoff, add `naip_2021` / `fl_naip_2021` means to `metadata/dataset_means.json`, run `Build_Data.py`.
