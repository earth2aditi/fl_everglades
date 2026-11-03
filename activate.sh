source "/scratch/ajasapar/deepbiosphere/miniforge3/etc/profile.d/conda.sh"
conda activate deepbio
export CONDA_PKGS_DIRS="/scratch/ajasapar/conda_pkgs" PIP_CACHE_DIR="/scratch/ajasapar/deepbiosphere/pip_cache" TMPDIR="/scratch/ajasapar/deepbiosphere/tmp"
export CUDA_HOME="$CONDA_PREFIX"
cd "/scratch/ajasapar/deepbiosphere/Deepbiosphere"
