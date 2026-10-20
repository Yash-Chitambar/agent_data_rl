# OS layer for the repo: glibc, git, rsync, EGL (glvnd) and uv. The Python env itself is
# created by scripts/server/bootstrap.sh inside the bind-mounted repo (.venv, .python), so the
# image stays small and the env matches uv.lock exactly. CUDA comes from the torch/Warp wheels;
# the NVIDIA driver (and its EGL library) is injected at run time by the NVIDIA container
# toolkit. UNTESTED (not built on the local machine).
#
#   docker build -t peg-test-os scripts/server/container
#   docker run --rm -it --gpus all --user $(id -u):$(id -g) -e HOME=/work/peg_test/.cache/home \
#     -v $PWD:/work/peg_test -w /work/peg_test peg-test-os bash scripts/server/bootstrap.sh
# Always mount the repo at the same path (/work/peg_test): .venv points into .python by path.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git rsync tmux build-essential python3 \
        libglvnd0 libegl1 libgl1 libgles2 libopengl0 libglx0 \
    && rm -rf /var/lib/apt/lists/*
# Let libglvnd find the driver's EGL implementation (library injected by the toolkit)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN curl -LsSf https://astral.sh/uv/0.12.18/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
WORKDIR /work/peg_test
