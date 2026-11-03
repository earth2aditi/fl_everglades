#!/usr/bin/env bash
# Set up Deepbiosphere in BlueHive 3 scratch, reusing packages already downloaded
# to /scratch/ajasapar/conda_pkgs.
# Run inside an interactive GPU job (not on the login node), e.g.:
#   srun -p gpu --gres=gpu:1 -c 8 --mem=32G -t 2:00:00 --pty bash
#   bash setup_deepbiosphere.sh
set -eo pipefail

# ---------------- edit these if needed ----------------
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/ajasapar}"
BASE="$SCRATCH_ROOT/deepbiosphere"
PKGS="$SCRATCH_ROOT/conda_pkgs"                 # existing package cache, reused as-is
ENV_NAME="deepbio"
PY_VER="3.10"
TORCH_VER="2.3.1"; TV_VER="0.18.1"; CUDA_TAG="cu121"; CUDA_FULL="12.1.1"
# GPU archs to compile inplace-abn for (V100=7.0, T4/RTX=7.5, A100=8.0, A40/A6000=8.6, H100=9.0)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;9.0}"
INSTALL_R="${INSTALL_R:-0}"   # 1 = also install R + rpy2 for the Maxent/RF baselines
# ------------------------------------------------------

mkdir -p "$BASE"/{pip_cache,tmp} "$PKGS"
export CONDA_PKGS_DIRS="$PKGS"
export PIP_CACHE_DIR="$BASE/pip_cache"
export TMPDIR="$BASE/tmp"

echo "Reusing package cache: $PKGS ($(ls "$PKGS" 2>/dev/null | wc -l) entries, $(du -sh "$PKGS" 2>/dev/null | cut -f1))"
df -h "$SCRATCH_ROOT" | tail -1

# 1. Miniforge in scratch (skipped if already installed)
if [ ! -x "$BASE/miniforge3/bin/conda" ]; then
  # reuse an installer left in the package dir or base dir if there is one
  INSTALLER=$(ls "$PKGS"/Miniforge3-Linux-x86_64.sh "$BASE"/miniforge.sh 2>/dev/null | head -1 || true)
  if [ -z "$INSTALLER" ]; then
    INSTALLER="$BASE/miniforge.sh"
    curl -L -o "$INSTALLER" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  fi
  bash "$INSTALLER" -b -p "$BASE/miniforge3"
fi
source "$BASE/miniforge3/etc/profile.d/conda.sh"
conda config --env --set pkgs_dirs "$PKGS" >/dev/null 2>&1 || true   # make the cache stick

# 2. Env with geo stack + compilers (cached packages are used automatically)
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" -c conda-forge \
    python="$PY_VER" numpy scipy pandas matplotlib seaborn scikit-learn tqdm requests \
    shapely rasterio geopandas pyproj descartes matplotlib-scalebar \
    tensorboard jupyter ninja gxx_linux-64=11 gcc_linux-64=11 git
fi
conda activate "$ENV_NAME"
conda install -y -c "nvidia/label/cuda-$CUDA_FULL" cuda-toolkit

# 3. PyTorch (pip wheel built against CUDA 12.1)
pip install "torch==$TORCH_VER" "torchvision==$TV_VER" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# 4. inplace-abn, compiled for the listed GPU archs
export CUDA_HOME="$CONDA_PREFIX"
export IABN_FORCE_CUDA=1
python -c "import inplace_abn" 2>/dev/null || \
  pip install --no-build-isolation inplace-abn || \
  pip install --no-build-isolation git+https://github.com/mapillary/inplace_abn.git

# 5. Optional R side for baselines
if [ "$INSTALL_R" = "1" ]; then
  conda install -y -c conda-forge r-base r-dismo r-randomforest r-reticulate r-rjava openjdk rpy2
fi

# 6. Clone + editable install
cd "$BASE"
[ -d Deepbiosphere ] || git clone https://github.com/moiexpositoalonsolab/Deepbiosphere.git
cd Deepbiosphere
pip install --no-build-isolation --no-deps -e .

# 7. Data directories + point Utils.py at them (also fixes the upstream stray-space bug)
DATA="$BASE/data"
mkdir -p "$DATA"/{occurrences,shpfiles,models,images,rasters,baselines,results,misc,docs,scratch,runs}
python - "$DATA" <<'EOF'
import re, sys, pathlib
data = sys.argv[1]
m = {"OCCS":"occurrences","SHPFILES":"shpfiles","MODELS":"models","IMAGES":"images",
     "RASTERS":"rasters","BASELINES":"baselines","RESULTS":"results","MISC":"misc",
     "DOCS":"docs","SCRATCH":"scratch","RUNS":"runs"}
f = pathlib.Path("src/deepbiosphere/Utils.py"); s = f.read_text()
s = re.sub(r"^[ \t]+paths = SimpleNamespace\(", "paths = SimpleNamespace(", s, flags=re.M)
for k, v in m.items():
    s = re.sub(rf"({k}\s*=\s*)'[^']*'", rf"\1'{data}/{v}/'", s, count=1)
f.write_text(s)
EOF

# 8. Activation helper for future sessions / sbatch scripts
cat > "$BASE/activate.sh" <<EOF
source "$BASE/miniforge3/etc/profile.d/conda.sh"
conda activate $ENV_NAME
export CONDA_PKGS_DIRS="$PKGS" PIP_CACHE_DIR="$BASE/pip_cache" TMPDIR="$BASE/tmp"
export CUDA_HOME="\$CONDA_PREFIX"
cd "$BASE/Deepbiosphere"
EOF

# 9. Smoke test
python - <<'EOF'
import torch, rasterio, geopandas, shapely, inplace_abn
import deepbiosphere.Utils as u, deepbiosphere.TResNet
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    from inplace_abn import InPlaceABN
    x = torch.randn(4, 8, 16, 16, device="cuda")
    InPlaceABN(8).cuda()(x); print("inplace-abn CUDA kernel OK on", torch.cuda.get_device_name(0))
print("IMAGES path ->", u.paths.IMAGES)
EOF
echo "Done. In future sessions / sbatch scripts:  source $BASE/activate.sh"
