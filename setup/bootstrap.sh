#!/usr/bin/env bash
# PlayVox bootstrap — any GPU server with a normal internet connection.
#
#   bash setup/bootstrap.sh --check    # 只体检，不改任何东西
#   bash setup/bootstrap.sh            # 缺什么补什么，然后启动
#   bash setup/bootstrap.sh --no-start # 只准备环境与资产
#
# 所有路径都可用环境变量覆盖，默认值：
#   PLAYVOX_WORKSPACE=$HOME/playvox
#   PLAYVOX_DATA=$WORKSPACE/assets
#   PLAYVOX_VENV=$WORKSPACE/venv
#   PLAYVOX_TORCH_CACHE / PLAYVOX_HF_CACHE = $WORKSPACE/cache/{torch,huggingface}
#   PLAYVOX_PORT=7871
#
# 资产来源（三选一，按优先级）：
#   PLAYVOX_ASSETS_DIR=<dir>               -> 从本地目录拷贝（离线/已下载）
#   PLAYVOX_MODEL_REPO=YuantaoChen/ForeHOI -> hf download（默认）
#   都不满足时脚本报错并提示先跑 setup/fetch_assets.sh
#
# 依赖策略：
#   * torch —— 已装则复用，否则装 2.3.1+cu121（与 nvcc 12.1 / spconv-cu121 对齐）。
#   * spconv-<cuda> / gsplat —— PyPI 预编译 wheel，不需要 nvcc；spconv 变体按
#     torch 实际编译的 CUDA 版本在运行时选择（cu118/cu121/cu124/cu126）。
#   * kaolin —— 已从 PlayVox 代码路径移除，不再需要。
#   * pytorch3d —— Linux 没有通用 wheel（官方索引只有 py310/cu121/torch2.4.1 的 0.7.8），
#     py311 必须源码编译。脚本先保证 nvcc 可用：优先系统 toolkit；没有就用
#     pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 并组一个 CUDA_HOME shim。
#     完全离线时用 PLAYVOX_PYTORCH3D_SRC=<tarball|wheel> 指向本地文件。
set -uo pipefail

WORK=${PLAYVOX_WORKSPACE:-$HOME/playvox}
DATA=${PLAYVOX_DATA:-$WORK/assets}
VENV=${PLAYVOX_VENV:-$WORK/venv}
TORCH_CACHE=${PLAYVOX_TORCH_CACHE:-$WORK/cache/torch}
HF_CACHE=${PLAYVOX_HF_CACHE:-$WORK/cache/huggingface}
PORT=${PLAYVOX_PORT:-7871}
REPO=$(cd "$(dirname "$0")/.." && pwd)
STAMP=$VENV/.playvox_deps_ok

MODE=run
case "${1:-}" in
  --check)    MODE=check ;;
  --no-start) MODE=prepare ;;
  -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
esac

log(){ printf '[playvox] %s\n' "$*"; }
die(){ printf '[playvox] ERROR: %s\n' "$*" >&2; exit 1; }

ASSETS=(checkpoints/forehoi.ckpt checkpoints/pipeline.yaml checkpoints/ss_generator.yaml \
        checkpoints/ss_decoder.ckpt checkpoints/ss_decoder.yaml \
        weights/sam2.1_hiera_large.pt)

missing_assets(){
  local miss=0 f
  for f in "${ASSETS[@]}"; do
    [ -s "$DATA/$f" ] || { log "missing $f"; miss=1; }
  done
  return $miss
}

# ── read-only health check ───────────────────────────────────────────────────
if [ "$MODE" = check ]; then
  log "workspace=$WORK data=$DATA venv=$VENV port=$PORT"
  command -v nvcc >/dev/null && log "nvcc: $(nvcc --version | tail -1)" || log "nvcc: ABSENT"
  "$VENV/bin/python" -V 2>/dev/null || log "venv: not built"
  if missing_assets; then log "assets: INCOMPLETE"; else log "assets: OK ($(du -sh "$DATA" | cut -f1))"; fi
  [ -f "$STAMP" ] && log "deps: stamped" || log "deps: not stamped"
  exit 0
fi

mkdir -p "$DATA"/{checkpoints,weights} "$TORCH_CACHE" "$HF_CACHE" "$(dirname "$VENV")"
export HF_HOME=$HF_CACHE TORCH_HOME=$TORCH_CACHE

# ── 1. venv ──────────────────────────────────────────────────────────────────
# 用完全隔离的 venv；没有 torch 时装 2.3.1（PyPI wheel 默认 cu121，与 nvcc 12.1 一致）。
# 不用 --system-site-packages：宿主环境里的旧包可能被判定为“已满足”而劫持依赖。
make_venv(){
  # 1) 标准 venv（本镜像 python3 -m venv 自带 pip 24.0）
  python3 -m venv "$VENV" 2>/dev/null && return 0
  # 2) virtualenv：自带 seed wheels，不依赖 ensurepip
  log "venv module unusable; falling back to virtualenv"
  { python3 -m pip install -q virtualenv \
      || python3 -m pip install -q --break-system-packages virtualenv; } 2>/dev/null || return 1
  python3 -m virtualenv -q "$VENV" && return 0
  return 1
}

if [ "${PLAYVOX_USE_SYSTEM_PYTHON:-0}" = 1 ]; then
  PY=$(command -v python3)
  STAMP=$WORK/.playvox_deps_system
  log "PLAYVOX_USE_SYSTEM_PYTHON=1 -> $PY"
else
  # 旧版脚本建的 venv 会继承系统包（含非法元数据的 pytorch-lightning），必须重建
  if [ -x "$VENV/bin/python" ] && grep -q 'include-system-site-packages = true' "$VENV/pyvenv.cfg" 2>/dev/null; then
    log "existing venv inherits system packages; recreating isolated venv"
    rm -rf "$VENV"
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    log "creating isolated venv at $VENV"
    make_venv || die "could not create a venv; retry with PLAYVOX_USE_SYSTEM_PYTHON=1"
  fi
  PY=$VENV/bin/python
  STAMP=$VENV/.playvox_deps_ok
fi
"$PY" -V >/dev/null 2>&1 || die "venv interpreter broken: $VENV"
log "python: $("$PY" -V 2>&1)"

# ── 2. assets ────────────────────────────────────────────────────────────────
# 只补缺失的文件。本地目录优先（PLAYVOX_ASSETS_DIR），否则从 HF 下载。
# HuggingFace 仓库里的路径是 <basename DATA>/checkpoints/...，所以 local_dir 用 DATA 的父目录。
repair_assets(){
  local missing=() f prefix
  for f in "${ASSETS[@]}"; do [ -s "$DATA/$f" ] || missing+=("$f"); done
  [ ${#missing[@]} -eq 0 ] && return 0
  log "incomplete (${#missing[@]}): ${missing[*]}"
  log "current $DATA/checkpoints:"
  ls -l "$DATA/checkpoints" 2>&1 | sed 's/^/[playvox]   /' || true

  if [ -n "${PLAYVOX_ASSETS_DIR:-}" ]; then
    log "copying assets from $PLAYVOX_ASSETS_DIR"
    cp -an "$PLAYVOX_ASSETS_DIR"/. "$DATA"/ || return 1
  else
    log "downloading from Hugging Face ${PLAYVOX_MODEL_REPO:-YuantaoChen/ForeHOI}"
    "$PY" -m pip -q install "huggingface_hub[cli,hf_transfer]" >/dev/null 2>&1
    HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" -m huggingface_hub.commands.huggingface_cli download \
      "${PLAYVOX_MODEL_REPO:-YuantaoChen/ForeHOI}" --repo-type model --local-dir "$DATA" \
      --include "${ASSETS[@]}" || return 1
  fi

  for f in "${ASSETS[@]}"; do
    [ -s "$DATA/$f" ] || { log "still missing: $f"; return 1; }
  done
}

repair_assets || die "assets incomplete under $DATA"

# MoGe prior: the pipeline reads its path from pipeline.yaml as-is. If the yaml
# points at a local file, make sure it is there; if it points at a repo id, the
# model is fetched at startup (needs network).
MOGE_FILE=$DATA/models/moge-vitl/model.pt
if [ -s "$MOGE_FILE" ]; then
  log "MoGe: local checkpoint present ($MOGE_FILE)"
else
  log "MoGe: $MOGE_FILE absent — pipeline.yaml must point at a valid path or repo id"
fi

# ── CUDA toolkit for source builds ───────────────────────────────────────────
# pytorch3d 在 Linux 上没有通用 wheel：官方索引只提供 py310_cu121_pyt241（0.7.8），
# py311 / 其它 torch 组合一律 403。所以 py311 镜像必须就地编译，先保证 nvcc 可用。
ensure_nvcc(){
  if [ -x "${CUDA_HOME:-}/bin/nvcc" ] && [ -f "${CUDA_HOME:-}/include/cuda_runtime.h" ]; then
    log "cuda toolkit: $("${CUDA_HOME}/bin/nvcc" --version | tail -1)"
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
    log "cuda toolkit: $(nvcc --version | tail -1)"
    return 0
  fi

  log "no system nvcc; installing nvcc + cuda headers via pip"
  $PIP install -q nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 || return 1
  local shim=$WORK/cuda-shim nv rt
  nv=$("$PY" -c 'import nvidia.cuda_nvcc,os;print(os.path.dirname(nvidia.cuda_nvcc.__file__))' 2>/dev/null) || return 1
  rt=$("$PY" -c 'import nvidia.cuda_runtime,os;print(os.path.dirname(nvidia.cuda_runtime.__file__))' 2>/dev/null) || return 1
  mkdir -p "$shim"
  ln -sfn "$nv/bin"     "$shim/bin"
  ln -sfn "$nv/nvvm"    "$shim/nvvm"
  ln -sfn "$rt/include" "$shim/include"
  ln -sfn "$rt/lib"     "$shim/lib64"
  export CUDA_HOME=$shim
  if [ -x "$shim/bin/nvcc" ] && "$shim/bin/nvcc" --version >/dev/null 2>&1 \
     && [ -f "$shim/include/cuda_runtime.h" ]; then
    log "cuda toolkit (pip shim): $("$shim/bin/nvcc" --version | tail -1)"
    return 0
  fi
  log "pip-provided nvcc shim unusable"
  return 1
}

# ── 3. dependencies ──────────────────────────────────────────────────────────
# 依赖 stamp 与 requirements 内容绑定：改依赖就自动重装
REQ_HASH=$(md5sum "$REPO/requirements.txt" 2>/dev/null | cut -c1-8)
[ -n "$REQ_HASH" ] && STAMP=$VENV/.playvox_deps_$REQ_HASH

if [ ! -f "$STAMP" ]; then
  log "installing dependencies (首次较慢)"
  PIP="$PY -m pip"
  $PIP install -q --upgrade pip
  $PIP install -q setuptools wheel ninja packaging

  # torch：隔离 venv 里没有，必装。2.3.1 的 PyPI wheel 默认就是 cu121，
  # 与本机 nvcc 12.1 / spconv-cu121 对齐（2.5.1 起默认变 cu124）。
  "$PY" -c 'import torch' 2>/dev/null || \
    $PIP install -q ${PLAYVOX_TORCH_SPEC:-torch==2.3.1 torchvision==0.18.1} || die "torch install failed"
  TORCH_VER=$("$PY" -c 'import torch;print(torch.__version__)')
  log "torch: $TORCH_VER (cuda_build $("$PY" -c 'import torch;print(torch.version.cuda)'))"
  case "$TORCH_VER" in
    2.2.*|2.3.*|2.4.*|2.5.*) : ;;
    *) log "WARN: torch $TORCH_VER 未在此组合上验证过（已验证 2.3.1 / 2.5.1 + cu121）" ;;
  esac

  # 约束文件：阻止 pip 因为 sam2 的 torch>=2.5.1 之类的声明而替换已装好的 CUDA 版 torch
  "$PY" - > /tmp/playvox_constraints.txt <<'PY' || true
import torch
print(f"torch=={torch.__version__.split('+')[0]}")
try:
    import torchvision
    print(f"torchvision=={torchvision.__version__.split('+')[0]}")
except Exception:
    pass
PY
  $PIP install -q -c /tmp/playvox_constraints.txt -r "$REPO/requirements.txt" \
    || die "base requirements failed"

  # spconv：按 torch 实际编译的 CUDA 版本选变体，避免 cu121/cu124 错配
  TORCH_CU=$("$PY" -c 'import torch;print(torch.version.cuda or "")')
  case "$TORCH_CU" in
    11.8) SPV=spconv-cu118 ;;
    12.1) SPV=spconv-cu121 ;;
    12.4) SPV=spconv-cu124 ;;
    12.6) SPV=spconv-cu126 ;;
    *)    SPV="" ;;
  esac
  if [ -n "$SPV" ]; then
    log "spconv: $SPV (torch cuda $TORCH_CU)"
    "$PY" -c 'import spconv' 2>/dev/null || $PIP install -q "$SPV" || die "$SPV install failed"
  else
    log "WARN: 未识别的 torch CUDA 版本 '$TORCH_CU'，跳过 spconv 自动安装"
  fi

  # 源码/压缩包安装：GitHub tarball 顶层多一层目录，pip 无法直接当 sdist 装
  src_install(){ # src_install <path-or-url> <label>
    local src=$1 name=$2 dir
    if [ -d "$src" ]; then
      $PIP install -q --no-deps --no-build-isolation "$src" && return 0; return 1
    fi
    case "$src" in
      *.tar.gz|*.tgz|*.tar|*.zip)
        dir=$WORK/src/$name
        rm -rf "$dir"; mkdir -p "$dir"
        tar xf "$src" -C "$dir" --strip-components=1 || return 1
        $PIP install -q --no-deps --no-build-isolation "$dir" && return 0; return 1 ;;
      *) $PIP install -q --no-deps --no-build-isolation "$src" ;;
    esac
  }

  # pytorch3d：Linux 无通用 wheel，就地编译（py311 无官方 wheel）
  if ! "$PY" -c 'import pytorch3d' 2>/dev/null; then
    if [ -n "${PLAYVOX_PYTORCH3D_SRC:-}" ]; then
      log "pytorch3d from $PLAYVOX_PYTORCH3D_SRC"
      src_install "$PLAYVOX_PYTORCH3D_SRC" pytorch3d || die "pytorch3d install failed"
    elif ensure_nvcc; then
      # 只编当前设备的架构，编译时间大约减半；限制并行度避免 nvcc OOM
      BUILD_ARCH=$("$PY" - <<'PY'
import torch
print("%d.%d" % (torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (8, 6)))
PY
)
      export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$BUILD_ARCH}"
      export MAX_JOBS="${MAX_JOBS:-4}"
      log "building pytorch3d from source (CUDA_HOME=$CUDA_HOME, arch=$TORCH_CUDA_ARCH_LIST, jobs=$MAX_JOBS)"
      FORCE_CUDA=1 src_install \
        "${PLAYVOX_PYTORCH3D_GIT:-git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47}" \
        pytorch3d || die "pytorch3d build failed"
    elif "$PY" -c 'import sys, torch; sys.exit(0 if sys.version_info[:2]==(3,10) and torch.__version__.startswith("2.4.1") else 1)'; then
      log "using official pytorch3d 0.7.8 wheel (py310/cu121/torch2.4.1)"
      $PIP install -q https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl \
        || die "pytorch3d wheel install failed"
    else
      die "pytorch3d: no nvcc and no matching wheel. Install a CUDA toolkit, or pass PLAYVOX_PYTORCH3D_SRC=<dir|tarball|wheel>"
    fi
  fi

  # gsplat：纯 Python wheel，首次渲染时才 JIT（PlayVox 不渲染 Gaussian，不会编译）
  "$PY" -c 'import gsplat' 2>/dev/null || $PIP install -q gsplat || die "gsplat install failed"

  # sam2 / MoGe：GitHub 不可达时用本地源码；--no-deps 防止其 torch>=2.5.1 声明升级 torch
  if [ -n "${PLAYVOX_SAM2_SRC:-}" ]; then
    src_install "$PLAYVOX_SAM2_SRC" sam2 || die "sam2 install failed"
  else
    $PIP install -q --no-deps "git+https://github.com/facebookresearch/sam2.git" || die "sam2 install failed"
  fi
  if [ -n "${PLAYVOX_MOGE_SRC:-}" ]; then
    src_install "$PLAYVOX_MOGE_SRC" moge || die "moge install failed"
  else
    $PIP install -q --no-deps \
      "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
      || die "moge install failed"
  fi

  touch "$STAMP"
  log "dependencies installed"
else
  log "dependencies already stamped"
fi

[ "$MODE" = prepare ] && { log "prepare done (not starting)"; exit 0; }

# ── 3b. import self-check ────────────────────────────────────────────────────
# 静态扫描抓不全 Hydra 的 _target_ 动态 import，所以装完直接真跑一遍。
log "verifying stage-1 imports (this loads torch)"
if ! ( cd "$REPO" && "$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
mods = [
    "ui",
    "tools.stage1",
    "tools.pipeline",
    "forehoi.config.loader",
    "forehoi.pipeline.inference_pipeline",
    "forehoi.model.backbone.tdfy_dit.models",
]
bad = []
for m in mods:
    try:
        __import__(m)
        print(f"  ok   {m}", flush=True)
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {e}", flush=True)
        bad.append(m)
sys.exit(1 if bad else 0)
PY
); then
  die "import self-check failed — 把上面的 FAIL 行发出来即可定位"
fi

# ── 4. launch ────────────────────────────────────────────────────────────────
fuser -k "$PORT/tcp" 2>/dev/null || true
sleep 1
log "launching on 0.0.0.0:$PORT (log /tmp/playvox_app.log)"
cd "$REPO"
nohup env \
  PLAYVOX_WORKSPACE="$WORK" PLAYVOX_DATA="$DATA" \
  TORCH_HOME="$TORCH_CACHE" HF_HOME="$HF_CACHE" \
  PLAYVOX_PIPELINE_CONFIG="${PLAYVOX_PIPELINE_CONFIG:-$DATA/checkpoints/pipeline.yaml}" \
  GRADIO_SERVER_PORT="$PORT" GRADIO_SERVER_NAME=0.0.0.0 \
  CONDA_PREFIX="${CONDA_PREFIX:-/usr/local/cuda}" \
  "$PY" app.py >/tmp/playvox_app.log 2>&1 </dev/null &

for _ in $(seq 1 150); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
    log "UP -> http://127.0.0.1:$PORT"
    log "访问方式：ssh -N -L $PORT:127.0.0.1:$PORT <this-host> 然后打开 http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 2
done
log "TIMEOUT — 检查 /tmp/playvox_app.log"
exit 1
