# PlayVox 外部资产

`app.py` 只跑 stage-1，需要的全部外部文件如下（合计约 **6.6 GB**）。

## 前提

三份 yaml（`pipeline.yaml` / `ss_generator.yaml` / `ss_decoder.yaml`）里的 `_target_` 必须指向本仓库的包名 `forehoi.*`。资产由外部维护，本仓库不做改写。

## 必需

| 路径（相对 `PLAYVOX_DATA`） | 大小 | 用途 |
|---|---:|---|
| `checkpoints/forehoi.ckpt` | 4.34 GB | stage-1 权重：SS DiT + 多视角条件 + 掩码分支 |
| `checkpoints/ss_decoder.ckpt` | 148 MB | 形状 latent → 64³ 占据 |
| `checkpoints/pipeline.yaml` | 3.6 KB | pipeline 结构入口 |
| `checkpoints/ss_generator.yaml` | 5.3 KB | 主干/条件编码器结构 |
| `checkpoints/ss_decoder.yaml` | 244 B | 解码器结构（缺了会 FileNotFoundError） |
| `weights/sam2.1_hiera_large.pt` | 898 MB | 交互式手/物体掩码传播 |
| `models/moge-vitl/model.pt` | 1.26 GB | 点图条件（depth 版 checkpoint 需要） |

一条命令下载：

```bash
bash setup/fetch_assets.sh            # 默认 $HOME/playvox/assets
```

## 不需要

| 文件 | 原因 |
|---|---|
| `checkpoints/slat_generator.ckpt`、`slat_decoder_*.ckpt` | stage-2（mesh/Gaussian），PlayVox 不生成 |
| `checkpoints/ss_encoder.*` | 只在训练/编码路径使用 |
| `weights/DA3/`、`weights/foundationpose/` | 位姿估计，PlayVox 不使用 |

## `pipeline.yaml` 需要包含的键

构造器已收窄，`pipeline.yaml` 只应保留 stage-1 需要的键：

```yaml
_target_: forehoi.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap
ss_generator_config_path: ss_generator.yaml
ss_decoder_config_path: ss_decoder.yaml
ss_decoder_ckpt_path: ss_decoder.ckpt
ss_preprocessor:       # 原样保留，_target_ 改成 forehoi.*
  ...
depth_model:           # 原样保留，_target_ 改成 forehoi.*
  ...
# 可选（有默认值）：
# dtype, device, ss_inference_steps, ss_rescale_t, ss_cfg_strength,
# ss_cfg_interval, ss_cfg_strength_pm, layout_post_optimization_method
```

以下键已从构造器移除，**yaml 里如果还留着会 TypeError**：

`slat_*`、`ss_encoder_*`、`slat_mean`、`slat_std`、`pad_size`、`version`、
`decode_formats`、`ss_condition_input_mapping`、`slat_condition_input_mapping`、
`compile_model`、`rendering_engine`、`downsample_ss_dist`、`pose_decoder_name`

## 缓存

| 路径 | 来源 | 何时需要 |
|---|---|---|
| `<TORCH_HOME>/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth` | torch.hub | 默认自动下载；服务器访问不了 GitHub 时用 `PLAYVOX_PRELOAD_DINO=1` 预置 |

## 校验

```bash
bash setup/bootstrap.sh --check     # 只检查，不下载、不启动
```
