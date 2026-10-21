#!/usr/bin/env bash
# Build a pytorch3d wheel for PlayVox.
#
# 必须在**有网络 + 有 CUDA toolkit** 的机器上跑。用途：
#   * 目标机装不了源码编译依赖时，在别的机器上预编译 wheel 再拷过去
#   * 锁定一份可复用的 wheel，避免每次部署都重新编译
#
#   bash setup/build_pytorch3d_wheel.sh [outdir]
#
# 产出 wheel 的兼容性取决于本机的 Python 小版本和 torch 大版本：
#   cp311 + torch 2.5.x  ->  只能装到 cp311 + torch 2.5.x 的目标机上
# 先确认目标机的 python / torch 版本，再选在哪台机器构建。
set -euo pipefail

OUT=${1:-$PWD/wheels}
COMMIT=${PYTORCH3D_COMMIT:-75ebeeaea0908c5527e7b1e305fbc7681382db47}
ARCH=${TORCH_CUDA_ARCH_LIST:-8.6}     # A10 / 4090 都是 sm_86

python - <<'PY'
import sys, torch
print(f"building with python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"cuda {torch.version.cuda}")
PY

SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT
echo "cloning pytorch3d @ $COMMIT"
git clone --filter=blob:none --quiet https://github.com/facebookresearch/pytorch3d.git "$SRC/pytorch3d"
git -C "$SRC/pytorch3d" checkout --quiet "$COMMIT"

mkdir -p "$OUT"
echo "building wheel (TORCH_CUDA_ARCH_LIST=$ARCH) — 约 5-15 分钟"
TORCH_CUDA_ARCH_LIST="$ARCH" python -m pip wheel "$SRC/pytorch3d" \
  -w "$OUT" --no-deps --no-build-isolation

ls -lh "$OUT"/pytorch3d*.whl
echo
echo "下一步：把上面的 wheel 上传到目标机，然后"
echo "  PLAYVOX_PYTORCH3D_SRC=$OUT/pytorch3d-*.whl bash setup/bootstrap.sh"
