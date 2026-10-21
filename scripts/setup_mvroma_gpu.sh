#!/usr/bin/env bash
# Dedicated Linux CUDA workspace. Reuses a preinstalled compatible CUDA PyTorch.
set -euo pipefail
MVROMA_ROOT="${1:?Usage: setup_mvroma_gpu.sh /absolute/workspace}"
mkdir -p "$MVROMA_ROOT/vendor" "$MVROMA_ROOT/weights" "$MVROMA_ROOT/logs"
cd "$MVROMA_ROOT"
python -m venv --system-site-packages .venv
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'
if [ ! -d vendor/MV-RoMa ]; then git clone https://github.com/IceTea-CV/MV-RoMa.git vendor/MV-RoMa; fi
if [ ! -d vendor/UFM ]; then git clone --recursive https://github.com/UniFlowMatch/UFM.git vendor/UFM; fi
git -C vendor/MV-RoMa checkout acb09efb0212129ac191031f6e56f150524b304f
git -C vendor/UFM checkout ff78d6c3db807c91578acccb58ee67d4394d5e5f
git -C vendor/UFM/UniCeption checkout f839559
.venv/bin/pip install --no-cache-dir -r vendor/MV-RoMa/requirements.txt \
  pycolmap-cuda12==4.2.1 gdown jaxtyping scikit-learn flow_vis rerun-sdk minio torchmetrics termcolor
.venv/bin/pip install --no-cache-dir --no-deps -e vendor/UFM/UniCeption -e vendor/UFM
if [ ! -f weights/indoor_final.pth ]; then
  .venv/bin/gdown 1L1GrTDy9N5u4CXR09fWOoAdCaQzgkeLV -O weights/indoor_final.pth
fi
.venv/bin/pip freeze > logs/environment.txt
.venv/bin/python -c 'import pycolmap; assert pycolmap.has_cuda; print(pycolmap.__version__)'
