#!/usr/bin/env bash
# 下载 PlayVox stage-1 需要的全部外部文件（普通服务器版，直连 HuggingFace）。
#
#   bash setup/fetch_assets.sh [目标目录]        # 默认 $HOME/playvox/assets
#   HF_ENDPOINT=https://hf-mirror.com bash setup/fetch_assets.sh   # 国内镜像
#
# 产物结构（与 bootstrap.sh 的 PLAYVOX_DATA 对齐）：
#   <dir>/checkpoints/forehoi.ckpt        4.3 GB  stage-1 权重
#   <dir>/checkpoints/ss_decoder.ckpt     148 MB  64^3 占据解码器
#   <dir>/checkpoints/{pipeline,ss_generator,ss_decoder}.yaml
#   <dir>/weights/sam2.1_hiera_large.pt   898 MB  交互式掩码
#   <dir>/models/moge-vitl/model.pt       1.3 GB  点图条件（depth 版 checkpoint 需要）
#
# DINOv2 默认交给 torch.hub 在首次启动时下载；服务器访问不了 GitHub 时用
#   PLAYVOX_PRELOAD_DINO=1 bash setup/fetch_assets.sh
# 预置到 $PLAYVOX_TORCH_CACHE/hub。
set -uo pipefail

D=${1:-${PLAYVOX_DATA:-$HOME/playvox/assets}}
TORCH_CACHE=${PLAYVOX_TORCH_CACHE:-$(dirname "$D")/cache/torch}
mkdir -p "$D"/{checkpoints,weights,models/moge-vitl}
cd "$D" || exit 1

get(){ # get <url> <out>
  local url=$1 out=$2
  if [ -s "$out" ]; then echo "skip  $(basename "$out")"; return 0; fi
  echo "fetch $(basename "$out")"
  curl -fL --retry 3 --retry-delay 2 --continue-at - -o "$out" "$url" || {
    echo "FAILED $url" >&2; return 1; }
}

HF=${HF_ENDPOINT:-https://huggingface.co}
BASE=$HF/YuantaoChen/ForeHOI/resolve/main
for f in checkpoints/forehoi.ckpt checkpoints/ss_decoder.ckpt \
         checkpoints/pipeline.yaml checkpoints/ss_generator.yaml \
         checkpoints/ss_decoder.yaml weights/sam2.1_hiera_large.pt; do
  get "$BASE/$f" "$f" || exit 1
done

get "$HF"/Ruicheng/moge-vitl/resolve/main/model.pt \
    models/moge-vitl/model.pt || exit 1

if [ "${PLAYVOX_PRELOAD_DINO:-0}" = 1 ]; then
  echo "=== DINOv2 (torch.hub offline preload) ==="
  mkdir -p "$TORCH_CACHE/hub/checkpoints"
  get https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth \
      "$TORCH_CACHE/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth" || exit 1
  if [ ! -d "$TORCH_CACHE/hub/facebookresearch_dinov2_main" ]; then
    curl -fL --retry 3 -o /tmp/dinov2-main.tar.gz \
      https://codeload.github.com/facebookresearch/dinov2/tar.gz/refs/heads/main || exit 1
    mkdir -p "$TORCH_CACHE/hub/facebookresearch_dinov2_main"
    tar xzf /tmp/dinov2-main.tar.gz -C "$TORCH_CACHE/hub/facebookresearch_dinov2_main" --strip-components=1
  fi
fi

echo "=== 结果 ==="
du -sh "$D"
du -sh "$D"/* 2>/dev/null
