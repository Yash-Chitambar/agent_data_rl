#!/usr/bin/env bash
# FoundationPose + SAM 3 env for the ZED pose estimator (docs/real/FOUNDATIONPOSE_20261116.md). Separate from every repo
# venv. Tested on mason (V100 sm_70, driver 580, gcc 13, CUDA toolkit 12.5 for the extensions), 2026-11-16.
# Usage: bash scripts/perception/zed_fp/install_fp_env.sh <work dir> [<conda root>] [<cuda home>] [<torch arch list>]
#   e.g.  bash scripts/perception/zed_fp/install_fp_env.sh $HOME/fp_work $HOME/miniforge3 /usr/local/cuda-12.5 "7.0"
#   (RTX 30xx: "8.6", RTX 40xx: "8.9"). SAM 3 is gated: run `hf auth login` (or put a read token in
#   ~/.cache/huggingface/token) with an account that has access to facebook/sam3 first.
# pytorch3d: FoundationPose only calls so3_exp_map / rotation_6d_to_matrix. The full build took about 30 min on mason;
# SKIP_P3D=1 uses scripts/perception/zed_fp/p3d_stub instead (same formulas; fp_python adds it when
# pytorch3d is missing).
set -euo pipefail
W=${1:?work dir}; CONDA=${2:-$HOME/miniforge3}; CUDA_HOME=${3:-/usr/local/cuda-12.5}; ARCH=${4:-7.0}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$W" && cd "$W"
"$CONDA/bin/mamba" create -y -n fp python=3.12
E="$CONDA/envs/fp"; P="$E/bin/pip"
"$P" install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$P" install "numpy<2" "setuptools<80" scipy scikit-learn h5py joblib PyYAML ruamel.yaml opencv-python-headless imageio \
  open3d trimesh transformations matplotlib pandas pyrender PyOpenGL kornia omegaconf psutil tqdm scikit-image ninja \
  pybind11 gdown huggingface_hub timm ftfy==6.1.1 regex iopath einops pycocotools
# Warp: the CUDA 12 build (the PyPI default is CUDA 13 and refuses sm_70)
"$P" install "https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl"
"$CONDA/bin/mamba" install -y -n fp -c conda-forge libusb          # open3d needs libusb-1.0.so.0
[ -d FoundationPose ] || git clone https://github.com/NVlabs/FoundationPose.git && git -C FoundationPose checkout a1b694b83e633c2cb6115b9063d940a687759392
[ -d nvdiffrast ] || git clone https://github.com/NVlabs/nvdiffrast.git && git -C nvdiffrast checkout 253ac4fcea7de5f396371124af597e6cc957bfae
[ -d sam3 ] || git clone https://github.com/facebookresearch/sam3.git && git -C sam3 checkout eef6839bcfc3458dd52ea80535dc6bd62aea86b1
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" FORCE_CUDA=1 MAX_JOBS=16
"$P" install --no-build-isolation -e ./sam3
"$P" install --no-build-isolation ./nvdiffrast
if [ "${SKIP_P3D:-0}" != 1 ]; then "$P" install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"; fi
# FoundationPose weights (Google Drive folder of the FoundationPose readme): refiner + scorer
mkdir -p weights_dl && (cd weights_dl && "$E/bin/gdown" --folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i)
mkdir -p FoundationPose/weights && cp -r weights_dl/no_diffusion/* FoundationPose/weights/
sha256sum -c - <<SHA
774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60  FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26  FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
SHA
# SAM 3 image model (gated)
"$E/bin/hf" download facebook/sam3 sam3.pt config.json --local-dir sam3_ckpt
echo "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e  sam3_ckpt/sam3.pt" | sha256sum -c -
# runner
cat > fp_python <<RUN
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$E/lib:\${LD_LIBRARY_PATH:-}"
export FP_ROOT="\${FP_ROOT:-$W/FoundationPose}" SAM3_CKPT="\${SAM3_CKPT:-$W/sam3_ckpt/sam3.pt}"
if ! "$E/bin/python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pytorch3d') else 1)" 2>/dev/null; then
  export PYTHONPATH="$REPO/scripts/perception/zed_fp/p3d_stub:\${PYTHONPATH:-}"
fi
exec "$E/bin/python" "\$@"
RUN
chmod +x fp_python
echo "done: $W/fp_python"
