"""PlayVox pipeline builder: ForeHOI stage-1 only.

Loads the sparse-structure generator, the ForeHOI stage-1 modules (multiview
conditioning, mask branch, gates), the sparse-structure decoder, DINOv2 and —
for the depth-conditioned checkpoint — MoGe.

The pipeline class builds the module structure only; the actual weights are
loaded from ``forehoi.ckpt`` here.
"""
from __future__ import annotations

import os

# Must be set before `forehoi` is imported: the package skips its heavy init and
# `core/inference.py` used to set CUDA_HOME the same way.
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
os.environ["CUDA_HOME"] = (
    os.environ.get("CONDA_PREFIX")
    or os.environ.get("CUDA_HOME")
    or "/usr/local/cuda"
)

import math

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from forehoi.config.loader import build_from_config
from forehoi.model.backbone.tdfy_dit.models.mask_branch import MaskBranch
from forehoi.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import (
    ModulatedMultiViewCond,
)
from tools import PIPELINE_CONFIG, stage1 as C


def build_pipeline(ckpt: str, device: str = "cuda", config_path: str | None = None):
    config_path = config_path or PIPELINE_CONFIG
    if not config_path or not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"pipeline.yaml not found: {config_path!r}. "
            "Set PLAYVOX_DATA (see setup/ASSETS.md) or pass config_path explicitly."
        )
    pipeline = build_from_config(config_path)
    states = torch.load(ckpt, map_location="cpu", weights_only=False)
    states = states.get("state_dict", states)
    arch = C.detect_arch(ckpt)
    ctx, use_depth = arch["cond_ctx"], arch["use_depth"]
    has_lora = any("lora_" in key for key in states)

    if has_lora:
        pipeline.models["ss_generator"] = get_peft_model(
            pipeline.models["ss_generator"],
            LoraConfig(r=256, lora_alpha=512, lora_dropout=0.0, target_modules=C.LORA_TARGETS),
        )
    pipeline.models["ss_condition_embedder"] = pipeline.condition_embedders["ss_condition_embedder"]

    dev = next(pipeline.models["ss_generator"].parameters()).device
    pipeline.models["multiview_cond"] = ModulatedMultiViewCond(
        1024, ctx, num_heads=16, mlp_ratio=4, attn_mode="full",
        use_checkpoint=False, use_rope=False, share_mod=False,
        qk_rms_norm=True, qk_rms_norm_cross=False,
    ).to(dev)
    ref_src = nn.Module()
    ref_src.weight = nn.Parameter(torch.empty(2, ctx))
    nn.init.normal_(ref_src.weight, std=1.0 / math.sqrt(ctx))
    pipeline.models["ref_src_emb"] = ref_src.to(dev)
    pipeline.models["mask_branch"] = MaskBranch(model_channels=1024, mask_res=C.MASK_RES, patch=4).to(dev)
    raw_gate = nn.Module()
    raw_gate.weight = nn.Parameter(torch.zeros(1))
    pipeline.models["raw_gate"] = raw_gate.to(dev)

    prefixes = [
        ("models.ss_generator.", "ss_generator"),
        ("models.multiview_cond.", "multiview_cond"),
        ("models.ref_src_emb.", "ref_src_emb"),
        ("models.mask_branch.", "mask_branch"),
        ("models.raw_gate.", "raw_gate"),
    ]
    if use_depth:
        pipeline.use_depth = True
        pipeline.cond_ctx_channels = ctx
        pipeline.models["ss_condition_embedder"].module_list[2].input_size = 296
        pointmap_gate = nn.Module()
        pointmap_gate.weight = nn.Parameter(torch.zeros(1))
        pipeline.models["pointmap_gate"] = pointmap_gate.to(dev)
        prefixes += [
            ("models.pointmap_gate.", "pointmap_gate"),
            ("models.ss_condition_embedder.", "ss_condition_embedder"),
        ]
    else:
        pipeline.use_depth = False
        pipeline.cond_ctx_channels = ctx

    # `ss_condition_embedder` intentionally reports ~700 missing keys: its two
    # DINO backbones come from torch.hub (frozen during training) and the
    # image/mask projection nets are not used by the manual build_cond path.
    # The pointmap stream (module_list.2 + projection_nets.2) is fully loaded.
    for prefix, key in prefixes:
        sub = {k.replace(prefix, ""): v for k, v in states.items() if k.startswith(prefix)}
        if sub:
            missing, unexpected = pipeline.models[key].load_state_dict(sub, strict=False)
            print(f"[PlayVox] {key}: loaded={len(sub)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    pipeline.models["ss_generator"].eval()
    print(
        f"[PlayVox] stage1-only pipeline ready: arch_ctx={ctx}, use_depth={use_depth}, "
        f"base_slats=not_loaded, ckpt={os.path.basename(ckpt)}",
        flush=True,
    )
    return pipeline
